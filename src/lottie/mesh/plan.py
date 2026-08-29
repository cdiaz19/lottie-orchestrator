"""Recorded execution plans and deterministic replay (E6).

A `Plan` is an execution **artifact**, not a prediction. The V3 spec asked for a DAG that a
mesh compiles to ahead of time, but a mesh routes dynamically by design — the supervisor
decides step N+1 from the result of step N — so there is nothing to compile in advance.
What IS reproducible is what a run actually decided, and that is what this records.

Replay drops into the existing engines unchanged: `MeshEngine.run` already takes `route`
as a parameter, so a recorded plan is just a different `RouteFn`. No engine changes.
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

from pydantic import BaseModel

from lottie.mesh.engine import RouteFn
from lottie.mesh.errors import MeshError
from lottie.mesh.schema import FINISH, MeshState, RouteDecision

#: Thread ids and agent names reach `plan_path` from CLI arguments, and
#: `Path(base) / "../../etc"` silently escapes — the hole PR #35 caught in the
#: distill drafts and S5b caught in sessions. Validated before any path join.
_THREAD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class PlanNotFound(MeshError):
    """No recorded plan exists for the requested run."""


class PlanDivergence(MeshError):
    """A replay asked for something the current mesh cannot do.

    Raised rather than skipped: a replay that quietly diverges from its recording is
    worse than one that refuses, because it looks like a reproduction and is not.
    """


class PlanStep(BaseModel):
    """One routing decision. More than one worker means a parallel fan-out."""

    step: int
    workers: list[str]


class Plan(BaseModel):
    """The routing decisions a run actually made, in order."""

    #: The task is stored as a HASH, never as text. A plan lives on disk, and the same
    #: discipline that keeps raw content out of the audit ledger applies here — a plan
    #: should not become a new place task text accumulates.
    task_sha256: str
    steps: list[PlanStep] = []
    created_at: float | None = None

    def matches(self, task: str) -> bool:
        """True when this plan was recorded for `task`."""
        return self.task_sha256 == hash_task(task)

    @property
    def worker_names(self) -> set[str]:
        return {w for s in self.steps for w in s.workers}


def hash_task(task: str) -> str:
    return hashlib.sha256(task.encode()).hexdigest()


class PlanRecorder:
    """Wraps a `RouteFn` and records what it decided.

    Records the DECISIONS as they happen rather than inferring them from
    `MeshState.history` afterwards. The inference approach looked cheaper — group history
    by `StepResult.step` — but it is wrong: only `LangGraphEngine` populates `step` (it
    needs it for deterministic parallel merge). `LocalEngine` leaves it at 0, so every
    sequential step would collapse into one phantom fan-out.

    Recording at the routing seam is exact by construction and identical across engines,
    which is also why no engine had to change.
    """

    def __init__(self, route: RouteFn) -> None:
        self._route = route
        self.decisions: list[list[str]] = []

    def __call__(self, state: MeshState) -> RouteDecision:
        decision = self._route(state)
        if decision.parallel:
            self.decisions.append(list(decision.parallel))
        elif decision.next != FINISH:
            self.decisions.append([decision.next])
        return decision

    def plan(self, task: str) -> Plan:
        return Plan(
            task_sha256=hash_task(task),
            steps=[
                PlanStep(step=index, workers=workers)
                for index, workers in enumerate(self.decisions)
            ],
            created_at=time.time(),
        )


def replay_route(plan: Plan, declared: set[str] | None = None) -> RouteFn:
    """Build a `RouteFn` that yields the recorded decisions instead of asking the LLM.

    Makes ZERO supervisor calls — that is the point of the slice, not a side effect.

    `declared` is the mesh's current worker set. When supplied, a plan naming a worker the
    mesh no longer declares raises `PlanDivergence` up front rather than failing obscurely
    mid-run, and never silently skips it.
    """
    if declared is not None:
        missing = sorted(plan.worker_names - declared)
        if missing:
            raise PlanDivergence(
                f"recorded plan needs worker(s) this mesh no longer declares: {missing}"
            )

    remaining = list(plan.steps)

    def route(state: MeshState) -> RouteDecision:
        if not remaining:
            return RouteDecision(next=FINISH)
        step = remaining.pop(0)
        if len(step.workers) > 1:
            # Same sentinel the supervisor uses for a fan-out: the engine reads
            # `parallel` first and ignores `next`.
            return RouteDecision(next=FINISH, parallel=list(step.workers))
        return RouteDecision(next=step.workers[0])

    return route


def plan_path(root: Path, agent: str, thread_id: str) -> Path:
    """Where a recorded plan lives. Thread ids come from the engine, but they reach here
    through a CLI argument, so they are validated before being joined onto a path."""
    if not _THREAD_RE.match(thread_id):
        raise PlanDivergence(f"invalid thread id {thread_id!r}")
    if not _THREAD_RE.match(agent):
        raise PlanDivergence(f"invalid agent name {agent!r}")
    return root / ".lottie" / "plans" / agent / f"{thread_id}.json"


def save_plan(root: Path, agent: str, thread_id: str, plan: Plan) -> Path:
    target = plan_path(root, agent, thread_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    return target


def load_plan(root: Path, agent: str, thread_id: str) -> Plan:
    target = plan_path(root, agent, thread_id)
    if not target.is_file():
        raise PlanNotFound(f"no recorded plan for {agent!r} thread {thread_id!r}")
    return Plan.model_validate_json(target.read_text(encoding="utf-8"))


def list_plans(root: Path, agent: str) -> list[str]:
    """Thread ids with a recorded plan, sorted."""
    base = root / ".lottie" / "plans" / agent
    if not base.is_dir():
        return []
    return sorted(p.stem for p in base.glob("*.json"))
