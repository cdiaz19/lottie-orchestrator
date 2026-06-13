# Phase 3 — Mesh Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adopt LangGraph as a real orchestration backend behind the existing `MeshEngine` ABC and deliver the three Phase-2-deferred capabilities — time-travel/replay (checkpointer), parallel fork/join, and human-in-the-loop interrupt/resume — while keeping the hand-rolled `LocalEngine` as the zero-dependency default.

**Architecture:** `langgraph` is an optional `[mesh]` extra. A new `LangGraphEngine(MeshEngine)` compiles a `StateGraph(MeshState)` (Pydantic state schema, `history` an `operator.add`-reduced channel), with a checkpointer (MemorySaver default, SqliteSaver for persistence). The engine ABC grows from "run to completion" to a `run(...) -> MeshRunResult` + `resume(...)` contract supporting pause/resume; `MeshOutput` gains back-compatible `status`/`thread_id`/`pending` fields. Parallel = supervisor fans out to a worker set (LangGraph `Send`), merged via the history reducer with a `StepResult.step` index for deterministic order. HITL = config `interrupt_before` workers compiled into LangGraph `interrupt_before`, surfaced as an interrupted result and continued via `MeshAgent.resume`.

**Tech Stack:** Python 3.12, Pydantic v2, **langgraph** (optional extra), Typer, pytest. `mypy --strict src` + `ruff check .` gate every file. Tests hermetic (MockLLMProvider, MemorySaver), skipif-guarded on langgraph.

---

## Spec reference

Implements `docs/superpowers/specs/2026-06-12-phase3-mesh-hardening-design.md`. Decisions D1–D9 are locked there; re-read before starting. This branch is `feat/phase3-mesh-hardening`, stacked on the Phase-2 mesh branch.

## Conventions (read once)

- Work from the project dir `/Users/cdiaz19/Documents/trae_projects/lottie-orchestrator` with venv active (`source .venv/bin/activate`).
- TDD throughout: failing test → red → implement → green → commit. One conventional commit per task.
- After every task: `mypy --strict src` and `ruff check .` clean before committing.
- **langgraph is optional.** Install for dev with `uv add --optional mesh langgraph` (Task 1 pins the version). All langgraph imports live ONLY in `src/lottie/mesh/langgraph_engine.py`, import-guarded. Tests that need it are decorated `@pytest.mark.skipif(not _HAS_LANGGRAPH, ...)`.
- Phase-2 types reused: `MeshState`, `StepResult`, `RouteDecision`, `MeshInput`, `MeshOutput`, `FINISH` (in `src/lottie/mesh/schema.py`); `MeshEngine`/`MeshNode`/`RouteFn` (`engine.py`); `LocalEngine` (`local.py`); `SupervisorRouter` (`router.py`); `MeshAgent` (`base.py`); `CapabilityViolation`/`MeshError`/`MeshStepLimitExceeded` (`errors.py`).

## File structure

**New:**
- `src/lottie/mesh/langgraph_engine.py` — `LangGraphEngine`, checkpointer factory, StateGraph build, parallel `Send`, `interrupt_before`, resume. ALL langgraph imports confined here.
- `src/lottie/mesh/checkpoint.py` — `build_checkpointer(kind, root)` → MemorySaver | SqliteSaver, import-guarded.
- `src/lottie/cli/mesh.py` — `lottie mesh resume|history` Typer sub-app.
- Tests: `src/lottie/mesh/tests/test_langgraph_engine.py`, `test_parallel.py`, `test_hitl.py`, `src/lottie/cli/tests/test_mesh_cli.py`, `tests/contracts/test_mesh_hardening_schema.py`.

**Modified:**
- `src/lottie/mesh/schema.py`, `engine.py`, `local.py`, `base.py`, `router.py`, `__init__.py`
- `src/lottie/project/config.py` (AgentConfig.interrupt_before)
- `src/lottie/serve/service.py`, `src/lottie/serve/schema.py` (interrupted status + resume_agent)
- `src/lottie/cli/app.py` (register mesh sub-app)
- `agents/assistant/` (third interrupt_before worker + config)
- `pyproject.toml` (`[mesh]` extra + mypy override), `.github/workflows/*` (install `.[mesh]`)

---

## Sub-phase A — LangGraphEngine foundation + checkpointer + time-travel

### Task 1: `[mesh]` extra + langgraph API spike

This task pins langgraph and **verifies the API assumptions** the rest of the plan depends on. The later LangGraph code is written against the standard API; if the pinned version differs, the findings here tell the implementer exactly what to adapt.

**Files:**
- Modify: `pyproject.toml`
- Create: `src/lottie/mesh/_langgraph_probe.py` (temporary spike script, deleted at end of task)

- [ ] **Step 1: Add the extra + mypy override.** In `pyproject.toml` under `[project.optional-dependencies]` add `mesh = ["langgraph>=0.2"]` (alongside `chroma`, `serve`). Under `[[tool.mypy.overrides]]` `module = [...]` add `"langgraph.*"` to the `ignore_missing_imports = true` list.

- [ ] **Step 2: Install it.** Run `uv add --optional mesh langgraph` (this resolves and pins the exact version into `uv.lock`). Then `python -c "import langgraph, importlib.metadata as m; print(m.version('langgraph'))"` and record the version.

- [ ] **Step 3: Verify the four API assumptions.** Write `src/lottie/mesh/_langgraph_probe.py` that proves each, printing PASS/FAIL:
```python
"""Temporary spike: verify the langgraph API the Phase-3 plan assumes. Deleted after Task 1."""

from __future__ import annotations

import operator
from typing import Annotated

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel


class S(BaseModel):
    task: str
    history: Annotated[list[str], operator.add] = []


def supervisor(state: S) -> dict[str, object]:
    return {}


def worker(state: S) -> dict[str, object]:
    return {"history": ["w"]}


def choose(state: S) -> str:
    return END if len(state.history) >= 1 else "worker"


def main() -> None:
    b = StateGraph(S)
    b.add_node("supervisor", supervisor)
    b.add_node("worker", worker)
    b.add_edge(START, "supervisor")
    b.add_conditional_edges("supervisor", choose, {"worker": "worker", END: END})
    b.add_edge("worker", "supervisor")
    g = b.compile(checkpointer=MemorySaver(), interrupt_before=["worker"])
    cfg = {"configurable": {"thread_id": "t1"}}
    out = g.invoke(S(task="x"), cfg)
    snap = g.get_state(cfg)
    print("PYDANTIC_STATE+REDUCER:", "PASS")
    print("INTERRUPT_BEFORE: next=", snap.next)  # expect ('worker',) → paused before worker
    resumed = g.invoke(None, cfg)                # resume from checkpoint
    print("RESUME: history=", resumed["history"] if isinstance(resumed, dict) else resumed.history)


if __name__ == "__main__":
    main()
```
Run `python src/lottie/mesh/_langgraph_probe.py`. Confirm: Pydantic state schema accepted, `operator.add` reducer appends, `interrupt_before` pauses (`snap.next` non-empty before the worker), and `invoke(None, cfg)` resumes. **Record the exact shapes** observed (does `invoke` return a dict or a model? what is `get_state(...).next`? how is the pending node read?) in the task's commit message body — Tasks 5–13 depend on these.

- [ ] **Step 4: If any assumption FAILS**, STOP and report which one, with the version and error. The fallback (per spec Risks) is a TypedDict state mirror; do not improvise — escalate so the plan's later tasks can be adjusted.

- [ ] **Step 5: Delete the probe + commit.** `rm src/lottie/mesh/_langgraph_probe.py`. Run `mypy --strict src` + `ruff check .` (clean — the probe is deleted so it's not linted; the only tracked change is `pyproject.toml` + `uv.lock`).
```bash
git add pyproject.toml uv.lock
git commit -m "build(mesh): add optional langgraph [mesh] extra; pin version after API spike"
```
Put the recorded API findings (return shapes, `get_state().next`, resume call) in the commit body.

### Task 2: Schema — history reducer + StepResult.step (back-compatible)

**Files:**
- Modify: `src/lottie/mesh/schema.py`
- Test: `src/lottie/mesh/tests/test_schema.py` (extend)

- [ ] **Step 1: Write failing tests** — append to `src/lottie/mesh/tests/test_schema.py`:
```python
def test_step_result_has_step_index_default_zero() -> None:
    from lottie.mesh.schema import StepResult

    assert StepResult(worker="w", result="r").step == 0
    assert StepResult(worker="w", result="r", step=3).step == 3


def test_mesh_state_history_still_appends_via_with_step() -> None:
    from lottie.mesh.schema import MeshState, StepResult

    s = MeshState(task="t").with_step(StepResult(worker="a", result="x"))
    assert [h.worker for h in s.history] == ["a"]
```

- [ ] **Step 2: Run, expect fail.** `pytest src/lottie/mesh/tests/test_schema.py::test_step_result_has_step_index_default_zero -v` → FAIL (`step` not a field).

- [ ] **Step 3: Implement.** In `src/lottie/mesh/schema.py`:
  - Add `import operator` and `from typing import Annotated` to the imports.
  - Add `step: int = 0` to `StepResult`.
  - Change `MeshState.history` to a reduced channel:
```python
class MeshState(BaseModel):
    """Evolving, typed state threaded through every mesh node."""

    task: str
    history: Annotated[list[StepResult], operator.add] = []
    final: str | None = None

    def with_step(self, step: StepResult) -> MeshState:
        """Return a new state with `step` appended (does not mutate self)."""
        return self.model_copy(update={"history": [*self.history, step]})
```
  The `Annotated[..., operator.add]` metadata is inert for plain Pydantic use (LocalEngine, `with_step`); LangGraph reads it as the channel reducer. `StepResult` is defined ABOVE `MeshState` already — keep that order.

- [ ] **Step 4: Run, expect pass.** `pytest src/lottie/mesh/tests/test_schema.py -v` → all pass (Phase-2 tests unaffected).

- [ ] **Step 5: Commit.**
```bash
git add src/lottie/mesh/schema.py src/lottie/mesh/tests/test_schema.py
git commit -m "feat(mesh): history reducer channel + StepResult.step index"
```

### Task 3: Schema — MeshRunResult, PendingApproval, ApprovalDecision, RouteDecision.parallel, MeshOutput fields

**Files:**
- Modify: `src/lottie/mesh/schema.py`
- Test: `src/lottie/mesh/tests/test_schema.py` (extend)

- [ ] **Step 1: Write failing tests** — append:
```python
def test_run_result_and_hitl_models() -> None:
    from lottie.mesh.schema import (
        ApprovalDecision,
        MeshRunResult,
        MeshState,
        PendingApproval,
    )

    r = MeshRunResult(state=MeshState(task="t"))
    assert r.status == "complete" and r.thread_id is None and r.pending is None
    p = PendingApproval(worker="deploy", proposed_input={"k": "v"})
    assert p.worker == "deploy"
    assert ApprovalDecision(action="approve").edited_input == {}
    assert ApprovalDecision(action="reject").action == "reject"


def test_route_decision_parallel_default_empty() -> None:
    from lottie.mesh.schema import RouteDecision

    assert RouteDecision(next="a").parallel == []
    assert RouteDecision(next="FINISH", parallel=["a", "b"]).parallel == ["a", "b"]


def test_mesh_output_status_defaults() -> None:
    from lottie.mesh.schema import MeshOutput

    o = MeshOutput(final="f")
    assert o.status == "complete" and o.thread_id is None and o.pending is None
```

- [ ] **Step 2: Run, expect fail.** `pytest src/lottie/mesh/tests/test_schema.py::test_run_result_and_hitl_models -v` → FAIL (import error).

- [ ] **Step 3: Implement.** In `src/lottie/mesh/schema.py` add `from typing import Annotated, Literal` (extend the existing typing import), and add these models. Add `parallel: list[str] = []` to `RouteDecision`, and the new fields to `MeshOutput`:
```python
class RouteDecision(BaseModel):
    """Supervisor's choice of the next worker(s), or FINISH."""

    next: str
    parallel: list[str] = []


class PendingApproval(BaseModel):
    """A worker about to run that requires human approval."""

    worker: str
    proposed_input: dict[str, str] = {}


class ApprovalDecision(BaseModel):
    """Human response to a pending approval."""

    action: Literal["approve", "reject"]
    edited_input: dict[str, str] = {}


class MeshRunResult(BaseModel):
    """Engine result: terminal state plus run status for pause/resume."""

    state: MeshState
    status: Literal["complete", "interrupted"] = "complete"
    thread_id: str | None = None
    pending: PendingApproval | None = None


class MeshOutput(BaseModel):
    """Output of a mesh agent."""

    final: str
    history: list[StepResult] = []
    status: Literal["complete", "interrupted"] = "complete"
    thread_id: str | None = None
    pending: PendingApproval | None = None
```

- [ ] **Step 4: Run, expect pass.** `pytest src/lottie/mesh/tests/test_schema.py -v`.

- [ ] **Step 5: Commit.**
```bash
git add src/lottie/mesh/schema.py src/lottie/mesh/tests/test_schema.py
git commit -m "feat(mesh): MeshRunResult/PendingApproval/ApprovalDecision + parallel/status fields"
```

### Task 4: Engine ABC + LocalEngine + MeshAgent contract change

Change the ABC to return `MeshRunResult` and add `resume`. Keep Phase-2 behavior intact via back-compatible unwrapping.

**Files:**
- Modify: `src/lottie/mesh/engine.py`, `src/lottie/mesh/local.py`, `src/lottie/mesh/base.py`
- Test: `src/lottie/mesh/tests/test_local_engine.py` (extend), `src/lottie/mesh/tests/test_mesh_agent.py` (extend)

- [ ] **Step 1: Write failing tests.** Append to `src/lottie/mesh/tests/test_local_engine.py`:
```python
def test_local_engine_run_returns_run_result() -> None:
    from lottie.mesh.local import LocalEngine
    from lottie.mesh.schema import FINISH, MeshRunResult, MeshState, RouteDecision

    def route(state: MeshState) -> RouteDecision:
        return RouteDecision(next=FINISH)

    result = LocalEngine().run(MeshState(task="t"), nodes={}, route=route, max_steps=8)
    assert isinstance(result, MeshRunResult)
    assert result.status == "complete"
    assert result.state.final == ""


def test_local_engine_resume_unsupported() -> None:
    import pytest

    from lottie.mesh.errors import MeshError
    from lottie.mesh.local import LocalEngine
    from lottie.mesh.schema import ApprovalDecision, MeshState, RouteDecision

    def route(state: MeshState) -> RouteDecision:
        return RouteDecision(next="FINISH")

    with pytest.raises(MeshError):
        LocalEngine().resume(
            "t1", nodes={}, route=route, decision=ApprovalDecision(action="approve")
        )
```
Append to `src/lottie/mesh/tests/test_mesh_agent.py`: confirm `_execute` still yields a `MeshOutput` with `status == "complete"`:
```python
def test_mesh_output_status_complete_on_normal_run() -> None:
    from lottie.llm import MockLLMProvider
    from lottie.mesh.base import MeshAgent
    from lottie.mesh.schema import MeshInput, MeshState, StepResult

    def node(state: MeshState) -> MeshState:
        return state.with_step(StepResult(worker="a", result="done"))

    mesh = MeshAgent(
        MockLLMProvider(["a", "FINISH"]),
        nodes={"a": node},
        descriptions={"a": "x"},
        enable_benchmarks=False,
    )
    out = mesh.run(MeshInput(task="t"))
    assert out.status == "complete" and out.thread_id is None and out.pending is None
    assert out.final == "done"
```

- [ ] **Step 2: Run, expect fail.** `pytest src/lottie/mesh/tests/test_local_engine.py -k run_result -v` → FAIL (run returns MeshState, not MeshRunResult).

- [ ] **Step 3: Implement.**
  `src/lottie/mesh/engine.py` — change the ABC (keep `MeshNode`/`RouteFn` aliases):
```python
from lottie.mesh.schema import MeshRunResult, MeshState, RouteDecision
# ... (ApprovalDecision import too)
from lottie.mesh.schema import ApprovalDecision

class MeshEngine(ABC):
    """Drives a supervisor→worker loop over typed state, with pause/resume."""

    @abstractmethod
    def run(
        self,
        initial: MeshState,
        *,
        nodes: Mapping[str, MeshNode],
        route: RouteFn,
        max_steps: int,
        thread_id: str | None = None,
    ) -> MeshRunResult:
        """Run until FINISH, max_steps, or an interrupt. Returns a MeshRunResult."""

    @abstractmethod
    def resume(
        self,
        thread_id: str,
        *,
        nodes: Mapping[str, MeshNode],
        route: RouteFn,
        decision: ApprovalDecision,
    ) -> MeshRunResult:
        """Continue an interrupted run from its checkpoint."""
```
  `src/lottie/mesh/local.py` — wrap the return and add `resume`:
```python
from lottie.mesh.errors import MeshError, MeshStepLimitExceeded
from lottie.mesh.schema import FINISH, ApprovalDecision, MeshRunResult, MeshState
# ...
class LocalEngine(MeshEngine):
    """Deterministic in-process supervisor loop (no checkpointing / HITL)."""

    def run(self, initial, *, nodes, route, max_steps, thread_id=None) -> MeshRunResult:
        state = initial
        for _ in range(max_steps):
            decision = route(state)
            if decision.next == FINISH:
                last = state.history[-1].result if state.history else ""
                return MeshRunResult(state=state.model_copy(update={"final": last}))
            state = nodes[decision.next](state)
        raise MeshStepLimitExceeded(
            f"routing loop exceeded max_steps={max_steps} without FINISH"
        )

    def resume(self, thread_id, *, nodes, route, decision) -> MeshRunResult:
        raise MeshError(
            "HITL resume requires LangGraphEngine; install lottie-orchestrator[mesh]"
        )
```
  (Keep the type annotations explicit on `run`/`resume` params to satisfy mypy --strict — mirror the ABC signature.)
  `src/lottie/mesh/base.py` — `_execute` unwraps `MeshRunResult` → `MeshOutput`:
```python
    def _execute(self, data: MeshInput) -> MeshOutput:
        def route(state: MeshState) -> RouteDecision:
            return self._router.route(state, self._descriptions)

        result = self._engine.run(
            MeshState(task=data.task),
            nodes=self._nodes,
            route=route,
            max_steps=data.max_steps,
        )
        return MeshOutput(
            final=result.state.final or "",
            history=result.state.history,
            status=result.status,
            thread_id=result.thread_id,
            pending=result.pending,
        )
```

- [ ] **Step 4: Run, expect pass.** `pytest src/lottie/mesh -v` → all pass (Phase-2 + new). If a Phase-2 test asserted `engine.run(...)` returns a `MeshState`, update it to read `.state` — those are the only expected breakages.

- [ ] **Step 5: Commit.**
```bash
git add src/lottie/mesh/engine.py src/lottie/mesh/local.py src/lottie/mesh/base.py src/lottie/mesh/tests/
git commit -m "feat(mesh): engine run→MeshRunResult + resume contract (LocalEngine wraps, rejects resume)"
```

### Task 5: LangGraphEngine.run (foundation) + checkpointer

> Write LangGraph code against the API confirmed in Task 1. If the pinned version's shapes differ from the snippets here (node return form, `get_state().next`, resume call), adapt to what Task 1 recorded — the SEMANTICS are fixed, the exact calls follow the installed version.

**Files:**
- Create: `src/lottie/mesh/checkpoint.py`, `src/lottie/mesh/langgraph_engine.py`
- Test: `src/lottie/mesh/tests/test_langgraph_engine.py`

- [ ] **Step 1: Write failing tests** (`src/lottie/mesh/tests/test_langgraph_engine.py`):
```python
from __future__ import annotations

import pytest

try:
    import langgraph  # noqa: F401

    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False

pytestmark = pytest.mark.skipif(not _HAS_LANGGRAPH, reason="needs [mesh] extra")

from collections.abc import Callable  # noqa: E402

from lottie.mesh.local import LocalEngine  # noqa: E402
from lottie.mesh.schema import (  # noqa: E402
    FINISH,
    MeshRunResult,
    MeshState,
    RouteDecision,
    StepResult,
)


def _node(name: str) -> Callable[[MeshState], MeshState]:
    def _run(state: MeshState) -> MeshState:
        return state.with_step(StepResult(worker=name, result=f"{name}:done"))

    return _run


def _scripted_route(script: list[str]) -> Callable[[MeshState], RouteDecision]:
    it = iter(script)

    def route(state: MeshState) -> RouteDecision:
        return RouteDecision(next=next(it))

    return route


def test_langgraph_engine_parity_with_local() -> None:
    from lottie.mesh.langgraph_engine import LangGraphEngine

    nodes = {"a": _node("a"), "b": _node("b")}
    lg = LangGraphEngine().run(
        MeshState(task="t"),
        nodes=nodes,
        route=_scripted_route(["a", "b", FINISH]),
        max_steps=8,
        thread_id="t1",
    )
    loc = LocalEngine().run(
        MeshState(task="t"), nodes=nodes, route=_scripted_route(["a", "b", FINISH]), max_steps=8
    )
    assert isinstance(lg, MeshRunResult) and lg.status == "complete"
    assert [s.worker for s in lg.state.history] == [s.worker for s in loc.state.history] == ["a", "b"]
    assert lg.state.final == loc.state.final == "b:done"
    assert lg.thread_id == "t1"
```

- [ ] **Step 2: Run, expect fail.** `pytest src/lottie/mesh/tests/test_langgraph_engine.py -v` → FAIL (no module).

- [ ] **Step 3: Implement.**
  `src/lottie/mesh/checkpoint.py`:
```python
"""Checkpointer factory for LangGraphEngine. langgraph imports are guarded."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from lottie.mesh.errors import MeshError

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


def build_checkpointer(kind: str = "memory", root: Path | None = None) -> BaseCheckpointSaver:
    """Return a MemorySaver (default) or a SqliteSaver under .lottie/mesh/."""
    try:
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise MeshError(
            "LangGraph mesh features require: pip install lottie-orchestrator[mesh]"
        ) from exc

    if kind == "memory":
        return MemorySaver()
    if kind == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        base = (root or Path.cwd()) / ".lottie" / "mesh"
        base.mkdir(parents=True, exist_ok=True)
        return SqliteSaver.from_conn_string(str(base / "checkpoints.db"))
    raise MeshError(f"unknown checkpointer kind {kind!r}")
```
  > Note: if Task 1 found `SqliteSaver.from_conn_string` is a context manager in the pinned version, adapt (e.g. store the cm and enter it) — record that in Task 1. For the MemorySaver default (all tests), this is straightforward.

  `src/lottie/mesh/langgraph_engine.py`:
```python
"""LangGraph-backed mesh engine. ALL langgraph imports are confined to this module."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from lottie.mesh.checkpoint import build_checkpointer
from lottie.mesh.engine import MeshEngine, MeshNode, RouteFn
from lottie.mesh.errors import MeshError
from lottie.mesh.schema import FINISH, ApprovalDecision, MeshRunResult, MeshState

try:
    from langgraph.graph import END, START, StateGraph

    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False


class LangGraphEngine(MeshEngine):
    """Runs the supervisor→worker mesh on a compiled LangGraph StateGraph."""

    def __init__(
        self,
        *,
        checkpoint: str = "memory",
        root: Path | None = None,
        interrupt_before: list[str] | None = None,
    ) -> None:
        if not _HAS_LANGGRAPH:
            raise MeshError(
                "LangGraphEngine requires: pip install lottie-orchestrator[mesh]"
            )
        self._checkpoint = checkpoint
        self._root = root
        self._interrupt_before = list(interrupt_before or [])

    def _build(self, nodes: Mapping[str, MeshNode], route: RouteFn) -> object:
        builder = StateGraph(MeshState)

        def supervisor(state: MeshState) -> dict[str, object]:
            return {}  # routing decision is computed in the conditional edge

        def choose(state: MeshState) -> str | list[str]:
            decision = route(state)
            if decision.parallel:
                return list(decision.parallel)  # fan-out (sub-phase B)
            return END if decision.next == FINISH else decision.next

        builder.add_node("supervisor", supervisor)
        for name, node in nodes.items():
            builder.add_node(name, self._wrap(node))
        builder.add_edge(START, "supervisor")
        path_map = {name: name for name in nodes}
        path_map[END] = END
        builder.add_conditional_edges("supervisor", choose, path_map)
        for name in nodes:
            builder.add_edge(name, "supervisor")

        saver = build_checkpointer(self._checkpoint, self._root)
        return builder.compile(
            checkpointer=saver,
            interrupt_before=self._interrupt_before or None,
        )

    @staticmethod
    def _wrap(node: MeshNode) -> object:
        def lg_node(state: MeshState) -> dict[str, object]:
            new_state = node(state)
            # return only the delta; the history reducer appends it
            delta = new_state.history[len(state.history):]
            return {"history": delta, "final": new_state.final}

        return lg_node

    def run(self, initial, *, nodes, route, max_steps, thread_id=None) -> MeshRunResult:
        graph = self._build(nodes, route)
        tid = thread_id or "default"
        config = {"configurable": {"thread_id": tid}, "recursion_limit": max_steps * 3}
        final = graph.invoke(initial, config)
        state = final if isinstance(final, MeshState) else MeshState.model_validate(final)
        last = state.history[-1].result if state.history else ""
        return MeshRunResult(
            state=state.model_copy(update={"final": state.final or last}),
            status="complete",
            thread_id=tid,
        )

    def resume(self, thread_id, *, nodes, route, decision) -> MeshRunResult:
        raise NotImplementedError  # implemented in sub-phase C
```
  > Adapt `final = graph.invoke(...)` handling and `recursion_limit` to the Task-1-recorded return shape. The parity test pins the behavior; tune the call until it passes. The `choose` parallel branch is inert until sub-phase B (route never sets `parallel` yet).

- [ ] **Step 4: Run, expect pass.** `pytest src/lottie/mesh/tests/test_langgraph_engine.py -v`. If the node-return form differs, adjust `_wrap`/`run` per Task 1 findings until parity holds.

- [ ] **Step 5: Commit.**
```bash
git add src/lottie/mesh/checkpoint.py src/lottie/mesh/langgraph_engine.py src/lottie/mesh/tests/test_langgraph_engine.py
git commit -m "feat(mesh): LangGraphEngine.run with checkpointer (parity with LocalEngine)"
```

### Task 6: Time-travel — checkpoint history/replay

**Files:**
- Modify: `src/lottie/mesh/langgraph_engine.py`
- Test: `src/lottie/mesh/tests/test_langgraph_engine.py` (extend)

- [ ] **Step 1: Write failing test** — append:
```python
def test_langgraph_engine_lists_checkpoint_history() -> None:
    from lottie.mesh.langgraph_engine import LangGraphEngine

    eng = LangGraphEngine()
    nodes = {"a": _node("a")}
    eng.run(MeshState(task="t"), nodes=nodes, route=_scripted_route(["a", FINISH]),
            max_steps=8, thread_id="hist1")
    snapshots = eng.history("hist1", nodes=nodes, route=_scripted_route([]))
    assert len(snapshots) >= 1
    # most recent snapshot reflects the completed run
    assert any(s.history for s in snapshots)
```

- [ ] **Step 2: Run, expect fail.** → `AttributeError: 'LangGraphEngine' object has no attribute 'history'`.

- [ ] **Step 3: Implement.** Add to `LangGraphEngine` a `history` method that rebuilds the graph and reads checkpoints:
```python
    def history(self, thread_id, *, nodes, route) -> list[MeshState]:
        """Return the MeshState at each persisted checkpoint for a thread (newest first)."""
        graph = self._build(nodes, route)
        config = {"configurable": {"thread_id": thread_id}}
        states: list[MeshState] = []
        for snap in graph.get_state_history(config):
            values = snap.values
            states.append(
                values if isinstance(values, MeshState) else MeshState.model_validate(values)
            )
        return states
```
  > `get_state_history` is the standard API; adapt the snapshot `.values` access per Task 1 findings if needed.

- [ ] **Step 4: Run, expect pass.** `pytest src/lottie/mesh/tests/test_langgraph_engine.py -v`.

- [ ] **Step 5: Commit.**
```bash
git add src/lottie/mesh/langgraph_engine.py src/lottie/mesh/tests/test_langgraph_engine.py
git commit -m "feat(mesh): LangGraphEngine checkpoint history (time-travel)"
```

---

## Sub-phase B — parallel fork/join

### Task 7: Router — parallel fan-out + StepResult.step stamping

**Files:**
- Modify: `src/lottie/mesh/router.py`
- Test: `src/lottie/mesh/tests/test_router.py` (extend)

- [ ] **Step 1: Write failing tests** — append to `src/lottie/mesh/tests/test_router.py`:
```python
def test_router_parses_parallel_fanout() -> None:
    # comma-separated names → parallel set; all must be declared
    router = SupervisorRouter(_complete_returning("research, critic"))
    decision = router.route(MeshState(task="t"), _WORKERS)
    assert sorted(decision.parallel) == ["critic", "research"]


def test_router_parallel_rejects_undeclared() -> None:
    router = SupervisorRouter(_complete_returning("research, hacker"))
    with pytest.raises(CapabilityViolation):
        router.route(MeshState(task="t"), _WORKERS)
```
  (`_complete_returning` and `_WORKERS` already exist in this file from Phase 2.)

- [ ] **Step 2: Run, expect fail.** `pytest src/lottie/mesh/tests/test_router.py -k parallel -v`.

- [ ] **Step 3: Implement.** In `src/lottie/mesh/router.py`, update `route`/`_resolve` to detect a comma-separated multi-worker reply and validate each name:
```python
    def route(self, state: MeshState, workers: Mapping[str, str]) -> RouteDecision:
        roster = "\n".join(f"- {name}: {desc}" for name, desc in workers.items())
        done = "\n".join(f"[{s.worker}] {s.result}" for s in state.history) or "(none yet)"
        user = (
            f"Task: {state.task}\n\n"
            f"Workers:\n{roster}\n\n"
            f"Work done so far:\n{done}\n\n"
            "Reply with ONE worker name, several comma-separated names to run them in "
            "parallel, or FINISH:"
        )
        raw = self._complete(
            [Message(role="system", content=_SYSTEM), Message(role="user", content=user)]
        ).content.strip()
        names = [p.strip() for p in raw.split(",") if p.strip()]
        if len(names) > 1:
            resolved = [self._resolve(n, workers) for n in names]
            return RouteDecision(next=FINISH, parallel=resolved)
        return RouteDecision(next=self._resolve(raw, workers))
```
  Update `_SYSTEM` to mention parallel:
```python
_SYSTEM = (
    "You are a supervisor routing a task to the best-suited worker(s). "
    "Reply with ONE worker name, several comma-separated names to run in parallel, "
    "or FINISH when the task is complete. Never reply with anything else."
)
```
  `_resolve` is unchanged (it already rejects undeclared names via `CapabilityViolation`).

  **StepResult.step:** the router does NOT set `step` (it doesn't know execution order). Parallel branches are genuinely concurrent, so there is no meaningful dispatch index to stamp; `step` defaults to `0` and the engine's deterministic merge falls back to sorting by `worker` name (Task 8). `step` remains an additive ordering hook for future use. No router change needed for `step`.

- [ ] **Step 4: Run, expect pass.** `pytest src/lottie/mesh/tests/test_router.py -v` (Phase-2 single-route tests still pass — single reply has no comma).

- [ ] **Step 5: Commit.**
```bash
git add src/lottie/mesh/router.py src/lottie/mesh/tests/test_router.py
git commit -m "feat(mesh): supervisor parallel fan-out (comma reply → validated worker set)"
```

### Task 8: LangGraphEngine parallel dispatch + deterministic merge

**Files:**
- Modify: `src/lottie/mesh/langgraph_engine.py`
- Test: `src/lottie/mesh/tests/test_parallel.py`

- [ ] **Step 1: Write failing test** (`src/lottie/mesh/tests/test_parallel.py`):
```python
from __future__ import annotations

import pytest

try:
    import langgraph  # noqa: F401

    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False

pytestmark = pytest.mark.skipif(not _HAS_LANGGRAPH, reason="needs [mesh] extra")

from collections.abc import Callable  # noqa: E402

from lottie.mesh.schema import FINISH, MeshState, RouteDecision, StepResult  # noqa: E402


def _node(name: str) -> Callable[[MeshState], MeshState]:
    def _run(state: MeshState) -> MeshState:
        return state.with_step(StepResult(worker=name, result=f"{name}:done"))

    return _run


def test_parallel_fanout_merges_all_workers() -> None:
    from lottie.mesh.langgraph_engine import LangGraphEngine

    # supervisor: fan out to a+b in parallel, then FINISH
    calls = iter([RouteDecision(next=FINISH, parallel=["a", "b"]), RouteDecision(next=FINISH)])

    def route(state: MeshState) -> RouteDecision:
        return next(calls)

    result = LangGraphEngine().run(
        MeshState(task="t"),
        nodes={"a": _node("a"), "b": _node("b")},
        route=route,
        max_steps=8,
        thread_id="par1",
    )
    workers = sorted(s.worker for s in result.state.history)
    assert workers == ["a", "b"]  # both ran, both merged
```

- [ ] **Step 2: Run, expect fail.** It will fail or hang if fan-out isn't handled; the `choose` function from Task 5 already returns `list(decision.parallel)` — verify whether the base LangGraph conditional-edge list-return fans out correctly, or whether `Send` is required.

- [ ] **Step 3: Implement.** If returning a list from the conditional edge does NOT fan out in the pinned version, switch `choose` to the `Send` API:
```python
        from langgraph.types import Send  # import at top of module, guarded with the others

        def choose(state: MeshState):  # type: ignore[no-untyped-def]
            decision = route(state)
            if decision.parallel:
                return [Send(name, state) for name in decision.parallel]
            return END if decision.next == FINISH else decision.next
```
  Keep `path_map` for the single-route case. The history reducer (`operator.add`) merges each branch's appended `StepResult`. After join, the supervisor runs again (next `route` call → FINISH).
  **Deterministic order:** sort `result.state.history` by `(step, worker)` before returning, so parallel completion order doesn't leak. Since `step` is `0` for all (concurrent branches), this is effectively a stable sort by `worker` name — deterministic regardless of which branch finishes first. Update `run`'s terminal handling:
```python
        ordered = sorted(state.history, key=lambda s: (s.step, s.worker))
        state = state.model_copy(update={"history": ordered, "final": state.final or last})
```
  Adapt to Task 1 findings on whether `Send` or list-return is correct for the pinned version.

- [ ] **Step 4: Run, expect pass.** `pytest src/lottie/mesh/tests/test_parallel.py -v`. Also re-run `test_langgraph_engine.py` to confirm single-route parity is intact.

- [ ] **Step 5: Commit.**
```bash
git add src/lottie/mesh/langgraph_engine.py src/lottie/mesh/tests/test_parallel.py
git commit -m "feat(mesh): LangGraphEngine parallel fan-out + deterministic history merge"
```

---

## Sub-phase C — HITL interrupt/resume

### Task 9: AgentConfig.interrupt_before + subset guard

**Files:**
- Modify: `src/lottie/project/config.py`
- Test: `src/lottie/mesh/tests/test_schema.py` (extend — config field)

- [ ] **Step 1: Write failing test** — append to `test_schema.py`:
```python
def test_agent_config_interrupt_before_field() -> None:
    from lottie.project.config import AgentConfig

    cfg = AgentConfig(provider="mock/x", workers=["a", "b"], interrupt_before=["b"])
    assert cfg.interrupt_before == ["b"]
    assert AgentConfig(provider="mock/x").interrupt_before == []
```

- [ ] **Step 2: Run, expect fail.** `pytest src/lottie/mesh/tests/test_schema.py::test_agent_config_interrupt_before_field -v`.

- [ ] **Step 3: Implement.** Add to `AgentConfig` in `src/lottie/project/config.py`:
```python
    workers: list[str] = []  # mesh routing allow-set (capability enforcement)
    interrupt_before: list[str] = []  # mesh workers that pause for human approval (HITL)
```

- [ ] **Step 4: Run, expect pass.**

- [ ] **Step 5: Commit.**
```bash
git add src/lottie/project/config.py src/lottie/mesh/tests/test_schema.py
git commit -m "feat(config): add interrupt_before allow-set to AgentConfig"
```

### Task 10: LangGraphEngine interrupt + resume

**Files:**
- Modify: `src/lottie/mesh/langgraph_engine.py`
- Test: `src/lottie/mesh/tests/test_hitl.py`

- [ ] **Step 1: Write failing test** (`src/lottie/mesh/tests/test_hitl.py`):
```python
from __future__ import annotations

import pytest

try:
    import langgraph  # noqa: F401

    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False

pytestmark = pytest.mark.skipif(not _HAS_LANGGRAPH, reason="needs [mesh] extra")

from collections.abc import Callable  # noqa: E402

from lottie.mesh.schema import (  # noqa: E402
    FINISH,
    ApprovalDecision,
    MeshState,
    RouteDecision,
    StepResult,
)


def _node(name: str) -> Callable[[MeshState], MeshState]:
    def _run(state: MeshState) -> MeshState:
        return state.with_step(StepResult(worker=name, result=f"{name}:done"))

    return _run


def test_interrupt_before_pauses_then_resume_approve_completes() -> None:
    from lottie.mesh.langgraph_engine import LangGraphEngine

    nodes = {"deploy": _node("deploy")}
    route = lambda s: RouteDecision(next="deploy") if not s.history else RouteDecision(next=FINISH)  # noqa: E731
    eng = LangGraphEngine(interrupt_before=["deploy"])

    paused = eng.run(MeshState(task="ship it"), nodes=nodes, route=route, max_steps=8, thread_id="h1")
    assert paused.status == "interrupted"
    assert paused.pending is not None and paused.pending.worker == "deploy"

    done = eng.resume("h1", nodes=nodes, route=route, decision=ApprovalDecision(action="approve"))
    assert done.status == "complete"
    assert [s.worker for s in done.state.history] == ["deploy"]


def test_resume_reject_skips_worker() -> None:
    from lottie.mesh.langgraph_engine import LangGraphEngine

    nodes = {"deploy": _node("deploy")}
    route = lambda s: RouteDecision(next="deploy") if not s.history else RouteDecision(next=FINISH)  # noqa: E731
    eng = LangGraphEngine(interrupt_before=["deploy"])
    eng.run(MeshState(task="ship it"), nodes=nodes, route=route, max_steps=8, thread_id="h2")
    done = eng.resume("h2", nodes=nodes, route=route, decision=ApprovalDecision(action="reject"))
    assert done.status == "complete"
    assert any(s.worker == "deploy" and "reject" in s.result.lower() for s in done.state.history)
```

- [ ] **Step 2: Run, expect fail.** `resume` raises `NotImplementedError`.

- [ ] **Step 3: Implement.** In `LangGraphEngine`:
  - `run`: after `graph.invoke(initial, config)`, check `graph.get_state(config)` — if `.next` is non-empty (paused before an interrupt node), return an interrupted result:
```python
    def run(self, initial, *, nodes, route, max_steps, thread_id=None) -> MeshRunResult:
        graph = self._build(nodes, route)
        tid = thread_id or "default"
        config = {"configurable": {"thread_id": tid}, "recursion_limit": max_steps * 3}
        graph.invoke(initial, config)
        return self._snapshot(graph, config, tid)

    def _snapshot(self, graph, config, tid) -> MeshRunResult:
        snap = graph.get_state(config)
        state = snap.values if isinstance(snap.values, MeshState) else MeshState.model_validate(snap.values)
        pending_nodes = [n for n in snap.next if n in self._interrupt_before]
        if pending_nodes:
            return MeshRunResult(
                state=state,
                status="interrupted",
                thread_id=tid,
                pending=PendingApproval(worker=pending_nodes[0], proposed_input={"task": state.task}),
            )
        last = state.history[-1].result if state.history else ""
        ordered = sorted(state.history, key=lambda s: (s.step, s.worker))
        return MeshRunResult(
            state=state.model_copy(update={"history": ordered, "final": state.final or last}),
            status="complete",
            thread_id=tid,
        )
```
  (Import `PendingApproval` from schema. Rebuild `graph` in `resume` the same way as `run` so the compiled graph + checkpointer share the thread.)
  - `resume`:
```python
    def resume(self, thread_id, *, nodes, route, decision) -> MeshRunResult:
        graph = self._build(nodes, route)
        config = {"configurable": {"thread_id": thread_id}}
        if decision.action == "reject":
            snap = graph.get_state(config)
            worker = snap.next[0] if snap.next else "unknown"
            # record the rejection and skip past the interrupt node
            graph.update_state(
                config,
                {"history": [StepResult(worker=worker, result="rejected by human")]},
                as_node=worker,
            )
        graph.invoke(None, config)  # continue from checkpoint
        return self._snapshot(graph, config, thread_id)
```
  > `update_state(..., as_node=...)` to advance past the interrupted node on reject is the standard API; adapt per Task 1 findings (the exact way to "skip" an interrupt node may differ — the semantics: on reject, do not run the worker, record a rejection StepResult, continue routing). If `as_node` skipping is unavailable in the pinned version, an acceptable alternative is to NOT invoke the node and instead resume routing by updating state with the rejection step and re-running the supervisor; document whichever the version supports.

- [ ] **Step 4: Run, expect pass.** `pytest src/lottie/mesh/tests/test_hitl.py -v`.

- [ ] **Step 5: Commit.**
```bash
git add src/lottie/mesh/langgraph_engine.py src/lottie/mesh/tests/test_hitl.py
git commit -m "feat(mesh): LangGraphEngine interrupt_before pause + approve/reject resume"
```

### Task 11: MeshAgent.resume + engine selection from config

**Files:**
- Modify: `src/lottie/mesh/base.py`
- Test: `src/lottie/mesh/tests/test_mesh_agent.py` (extend, skipif-guarded)

- [ ] **Step 1: Write failing test** — append to `test_mesh_agent.py` (guarded):
```python
def test_mesh_agent_resume_delegates_to_engine() -> None:
    import pytest

    try:
        import langgraph  # noqa: F401
    except ImportError:
        pytest.skip("needs [mesh] extra")

    from lottie.llm import MockLLMProvider
    from lottie.mesh.base import MeshAgent
    from lottie.mesh.langgraph_engine import LangGraphEngine
    from lottie.mesh.schema import ApprovalDecision, MeshInput, MeshState, StepResult

    def node(state: MeshState) -> MeshState:
        return state.with_step(StepResult(worker="deploy", result="shipped"))

    mesh = MeshAgent(
        MockLLMProvider(["deploy", "FINISH", "FINISH"]),
        nodes={"deploy": node},
        descriptions={"deploy": "ships"},
        engine=LangGraphEngine(interrupt_before=["deploy"]),
        enable_benchmarks=False,
    )
    out = mesh.run(MeshInput(task="ship"))
    assert out.status == "interrupted" and out.thread_id and out.pending
    resumed = mesh.resume(out.thread_id, ApprovalDecision(action="approve"))
    assert resumed.status == "complete"
    assert any(s.worker == "deploy" for s in resumed.history)
```

- [ ] **Step 2: Run, expect fail.** `MeshAgent` has no `resume`.

- [ ] **Step 3: Implement.** Add `resume` to `MeshAgent` in `src/lottie/mesh/base.py`. It must run through the same routing closure as `_execute`, and instrument like a run. Refactor the `route` closure into a helper and add `resume`:
```python
    def _route_fn(self):  # type: ignore[no-untyped-def]
        def route(state: MeshState) -> RouteDecision:
            return self._router.route(state, self._descriptions)
        return route

    def _to_output(self, result: MeshRunResult) -> MeshOutput:
        return MeshOutput(
            final=result.state.final or "",
            history=result.state.history,
            status=result.status,
            thread_id=result.thread_id,
            pending=result.pending,
        )

    def _execute(self, data: MeshInput) -> MeshOutput:
        result = self._engine.run(
            MeshState(task=data.task),
            nodes=self._nodes,
            route=self._route_fn(),
            max_steps=data.max_steps,
        )
        return self._to_output(result)

    def resume(self, thread_id: str, decision: ApprovalDecision) -> MeshOutput:
        """Continue an interrupted mesh run from its checkpoint."""
        result = self._engine.resume(
            thread_id, nodes=self._nodes, route=self._route_fn(), decision=decision
        )
        return self._to_output(result)
```
  Add imports for `MeshRunResult`, `ApprovalDecision`. (Note: `resume` is not wrapped by `BaseAgent.run`'s instrumentation; that's acceptable — resume metrics are a follow-up. Keep it simple.)

- [ ] **Step 4: Run, expect pass.** `pytest src/lottie/mesh/tests/test_mesh_agent.py -v`.

- [ ] **Step 5: Commit.**
```bash
git add src/lottie/mesh/base.py src/lottie/mesh/tests/test_mesh_agent.py
git commit -m "feat(mesh): MeshAgent.resume for HITL continuation"
```

### Task 12: Mesh package exports

**Files:**
- Modify: `src/lottie/mesh/__init__.py`
- Test: `src/lottie/mesh/tests/test_schema.py` (extend export check)

- [ ] **Step 1: Write failing test** — append:
```python
def test_phase3_exports() -> None:
    import lottie.mesh as m

    for sym in ("MeshRunResult", "PendingApproval", "ApprovalDecision", "build_checkpointer"):
        assert hasattr(m, sym), sym
```
  (`LangGraphEngine` is intentionally NOT eagerly exported from `lottie.mesh` to avoid importing langgraph at package import; it is imported directly from `lottie.mesh.langgraph_engine`. Do not add it to `__init__`.)

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Implement.** In `src/lottie/mesh/__init__.py` add to the imports + `__all__`: `MeshRunResult`, `PendingApproval`, `ApprovalDecision` (from `lottie.mesh.schema`) and `build_checkpointer` (from `lottie.mesh.checkpoint`). Do NOT import `langgraph_engine` here (keep langgraph out of the base import path).

- [ ] **Step 4: Run, expect pass.** `pytest src/lottie/mesh/tests/ -v`.

- [ ] **Step 5: Commit.**
```bash
git add src/lottie/mesh/__init__.py src/lottie/mesh/tests/test_schema.py
git commit -m "feat(mesh): export Phase-3 schema + checkpointer (LangGraphEngine stays lazy)"
```

### Task 13: AgentService — surface interrupted status + resume_agent

**Files:**
- Modify: `src/lottie/serve/schema.py`, `src/lottie/serve/service.py`
- Test: `src/lottie/serve/tests/test_service.py` (extend)

- [ ] **Step 1: Read the current `RunResult`** in `src/lottie/serve/schema.py` to confirm its fields (agent, output, latency_ms, input_tokens, output_tokens, cost_usd). **Write failing test** in `src/lottie/serve/tests/test_service.py` (skipif-guarded on langgraph) that builds a fixture project with an `interrupt_before` mesh, runs it via `AgentService.run_agent`, asserts the `RunResult` reports `status == "interrupted"` and a `thread_id`, then `AgentService.resume_agent(name, thread_id, ApprovalDecision(action="approve"))` returns `status == "complete"`. Model the fixture on `agents/research/tests/test_research_from_project.py` (mock embeddings env + copytree the units). Keep the mock LLM script sized to the routing+worker calls.

- [ ] **Step 2: Run, expect fail.** `RunResult` has no `status`.

- [ ] **Step 3: Implement.**
  `src/lottie/serve/schema.py` — add to `RunResult` (back-compatible defaults):
```python
    status: str = "complete"
    thread_id: str | None = None
    pending: dict[str, object] | None = None
```
  `src/lottie/serve/service.py` — in `run_agent`, after `output = agent.run(data)`, if `output` has a `status` attribute equal to `"interrupted"` (duck-typed; only meshes do), populate the new `RunResult` fields from `output.status/thread_id/pending`. Add `resume_agent`:
```python
    def resume_agent(self, name, thread_id, decision) -> RunResult:
        """Resume an interrupted mesh agent from its checkpoint."""
        # reconstruct the agent via the same DI seam as run_agent, then call .resume
        ...
```
  Reconstruct the agent exactly as `run_agent` does (load config, build provider, `instantiate_agent`), then call `agent.resume(thread_id, decision)`; map the returned `MeshOutput` to `RunResult` (same mapping as run). Raise `AgentExecutionError` if the agent has no `resume` (non-mesh).
  **Important:** with `MemorySaver` the checkpoint is process-local — `resume_agent` only works in the same process unless the mesh uses `SqliteSaver`. Document this in the method docstring (cross-process resume needs the sqlite checkpointer).

- [ ] **Step 4: Run, expect pass.** `pytest src/lottie/serve/tests/test_service.py -v` (and confirm the Phase-2 serve tests still pass — `status` defaults to `"complete"` for plain agents).

- [ ] **Step 5: Commit.**
```bash
git add src/lottie/serve/schema.py src/lottie/serve/service.py src/lottie/serve/tests/test_service.py
git commit -m "feat(serve): surface mesh interrupted status + resume_agent"
```

### Task 14: CLI — `lottie mesh resume|history`

**Files:**
- Create: `src/lottie/cli/mesh.py`
- Modify: `src/lottie/cli/app.py`
- Test: `src/lottie/cli/tests/test_mesh_cli.py`

- [ ] **Step 1: Read `src/lottie/cli/memory.py` or `knowledge.py`** for the Typer sub-app + `find_project_root` pattern. **Write failing test** (`src/lottie/cli/tests/test_mesh_cli.py`, skipif-guarded on langgraph) using `CliRunner`: in a fixture project with an `interrupt_before` mesh, `lottie run <mesh> --input '{"task":"x"}'` prints an "awaiting approval" line + a thread id; `lottie mesh resume <thread_id> --decision approve` (same process) completes. If cross-process resume isn't feasible in the test, assert at minimum that `lottie mesh history <thread_id>` and `lottie mesh resume --help` wire up (exit 0) — keep the test hermetic and deterministic.

- [ ] **Step 2: Run, expect fail.** No `mesh` command.

- [ ] **Step 3: Implement.** `src/lottie/cli/mesh.py` — a `typer.Typer()` sub-app with:
  - `resume(thread_id: str, decision: str = "approve", input: str = "")` → loads the project, finds the mesh (the run that produced the thread), calls `AgentService.resume_agent`, prints the result. Since a thread_id alone doesn't name the agent, require `--agent <name>` (add it as an option). Validate `decision ∈ {approve, reject}`.
  - `history(thread_id: str, agent: str)` → reconstructs the mesh and prints `LangGraphEngine.history(...)` snapshots (worker order per checkpoint) via `rich`.
  Both require the `[mesh]` extra; on `ImportError`/`MeshError`, print the install hint and exit non-zero.
  `src/lottie/cli/app.py` — `app.add_typer(mesh_app, name="mesh")`.

- [ ] **Step 4: Run, expect pass.** `pytest src/lottie/cli/tests/test_mesh_cli.py -v`.

- [ ] **Step 5: Commit.**
```bash
git add src/lottie/cli/mesh.py src/lottie/cli/app.py src/lottie/cli/tests/test_mesh_cli.py
git commit -m "feat(cli): lottie mesh resume/history"
```

### Task 15: Assistant mesh — HITL end-to-end with a third worker

**Files:**
- Modify: `agents/assistant/agent.py`, `agents/assistant/config.yaml`, `agents/assistant/AGENT.md`
- Create: a third worker `agents/publisher/` (minimal, like `critic`)
- Test: `agents/assistant/tests/test_assistant_hitl.py`

- [ ] **Step 1: Scaffold the worker + write failing test.** `lottie create agent publisher`, then replace its files with a minimal single-LLM-call worker (mirror `agents/critic/`): `PublisherInput{text}` / `PublisherOutput{published}`, one `self.complete`. Write `agents/assistant/tests/test_assistant_hitl.py` (skipif-guarded on langgraph): build `AssistantMesh.from_project` with mock embeddings, configure `interrupt_before: [publisher]` and a `LangGraphEngine`; run a task scripted to route research→publisher; assert `status == "interrupted"`, `pending.worker == "publisher"`; `mesh.resume(thread_id, approve)` completes with a publisher step.

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Implement.**
  - Create `agents/publisher/` (schema/prompts/agent/config.yaml/AGENT.md/__init__.py) mirroring `agents/critic/` exactly, renamed; output field `published`.
  - `agents/assistant/agent.py`: add `publisher` to `_DESCRIPTIONS` and wire a `_publisher_node` (mirror `_critic_node`, build `PublisherInput(text=last result)`, `_accumulate`, append `StepResult(worker="publisher", ...)`). In `from_project`, when constructing the `MeshAgent`, pass `engine=LangGraphEngine(interrupt_before=config.interrupt_before)` IF `config.interrupt_before` is non-empty AND langgraph is available; else keep the default `LocalEngine`. Guard the import. Extend the existing workers/descriptions consistency guard to also assert `set(config.interrupt_before) <= set(_DESCRIPTIONS)`.
  - `agents/assistant/config.yaml`: add `publisher` to `workers` and `interrupt_before: [publisher]`.
  - `agents/assistant/AGENT.md`: document the HITL worker + interrupt behavior.

- [ ] **Step 4: Run, expect pass.** `pytest agents/assistant -v agents/publisher -v`.

- [ ] **Step 5: Commit.**
```bash
git add agents/assistant/ agents/publisher/
git commit -m "feat(agents): assistant HITL — publisher worker with interrupt_before"
```

---

## Sub-phase D — contracts, gate, docs

### Task 16: Contract tests for Phase-3 schemas

**Files:**
- Create: `tests/contracts/test_mesh_hardening_schema.py`

- [ ] **Step 1: Write the tests** — round-trip `model_validate_json(model_dump_json())` equality for `MeshRunResult`, `PendingApproval`, `ApprovalDecision`, the extended `MeshOutput` (with status/thread_id/pending set), and `RouteDecision` (with `parallel`). Mirror `tests/contracts/test_mesh_schema.py` style.

- [ ] **Step 2: Run.** `pytest tests/contracts/test_mesh_hardening_schema.py -v` → pass.

- [ ] **Step 3: Commit.**
```bash
git add tests/contracts/test_mesh_hardening_schema.py
git commit -m "test(contracts): Phase-3 mesh schema round-trips"
```

### Task 17: Full gate + CI + docs sync

**Files:**
- Modify: `.github/workflows/*` (install `.[mesh]`), `CLAUDE.md`, `README.md`, `LOTTIE_PHASE0_SPEC.md`

- [ ] **Step 1: Run the full gate twice — with and without the extra.**
  - With extra (dev env has it): `pytest -q` → all pass (langgraph tests run). `pytest --cov=lottie --cov-report=term-missing -q` → coverage ≥ 80%. `mypy --strict src` clean. `ruff check .` clean.
  - Without extra (verify guards): `uv run --no-dev` is not it; instead confirm the skipif markers exist on every langgraph test (grep `_HAS_LANGGRAPH`) so a base install skips them cleanly. Confirm `python -c "import lottie.mesh"` works WITHOUT langgraph installed (the package import must not pull langgraph — that's why `__init__` doesn't import `langgraph_engine`).

- [ ] **Step 2: Update CI.** In `.github/workflows/*.yml`, change the install step to include the extra (e.g. `uv sync --extra mesh` or `pip install -e ".[mesh]"`) so langgraph tests run in CI. Keep one job (or matrix leg) WITHOUT the extra to prove the base install + skips stay green, if the CI is cheap to extend; otherwise note it.

- [ ] **Step 3: Update docs.**
  - `CLAUDE.md` mesh structure line: note `LangGraphEngine` (optional `[mesh]` extra), checkpointer, parallel + HITL.
  - `README.md` roadmap row `v0.4.0` → mesh hardening shipped (or 🚧 until merged); update Status line.
  - `LOTTIE_PHASE0_SPEC.md` release table `v0.4.0` row → mesh hardening (note governance shifted out).

- [ ] **Step 4: Commit.**
```bash
git add .github CLAUDE.md README.md LOTTIE_PHASE0_SPEC.md
git commit -m "docs+ci: close Phase 3 mesh hardening, run [mesh] extra in CI"
```

---

## Definition of done

- `pytest -q` green WITH `[mesh]` installed; coverage ≥ 80%; `mypy --strict src` + `ruff check .` clean.
- `python -c "import lottie.mesh"` works WITHOUT langgraph (package import never pulls it); langgraph tests skip cleanly on a base install.
- `LocalEngine` remains the default; `resume`/`interrupt_before` on `LocalEngine` raise a clear `MeshError`.
- `LangGraphEngine`: parity with LocalEngine on single-route; parallel fan-out merges deterministically (sorted by `step`, `worker`); `interrupt_before` pauses → `MeshAgent.resume(approve|reject)` continues.
- Time-travel: `LangGraphEngine.history(thread_id, ...)` lists checkpoints.
- `AgentService` surfaces interrupted status + `resume_agent`; `lottie mesh resume|history` wired.
- Phase-2 mesh + serve/run tests unchanged (back-compatible defaults).
- No langgraph in core `[project] dependencies` (only the `[mesh]` extra).
- Tag `v0.4.0` cut on merge to main (confirm before pushing, per the always-ask-before-push rule). This branch is stacked on `feat/phase2-agent-mesh`; merge that (PR #8) first, then this.

## Deferred (do NOT build here)

Policy-driven interrupts (Governance phase), supervisor-emitted dynamic INTERRUPT, parallel/HITL in LocalEngine, REST/web resume UI, skill-internal LLM token accounting (carried Phase-1 FU-2).
