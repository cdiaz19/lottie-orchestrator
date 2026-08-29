"""Recorded plans and deterministic replay (E6).

The load-bearing claim: replay re-executes a recorded run with ZERO supervisor calls.
That is what makes a multi-agent flow testable — today it is non-deterministic, so a
mesh test either hand-mocks the router or cannot assert on the path taken.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from lottie.mesh.local import LocalEngine
from lottie.mesh.plan import (
    Plan,
    PlanDivergence,
    PlanNotFound,
    PlanRecorder,
    PlanStep,
    hash_task,
    list_plans,
    load_plan,
    plan_path,
    replay_route,
    save_plan,
)
from lottie.mesh.schema import (
    FINISH,
    MeshState,
    RouteDecision,
    StepResult,
)


def _nodes(
    names: tuple[str, ...], visited: list[str] | None = None
) -> dict[str, Callable[[MeshState], MeshState]]:
    """Worker nodes that append their own step, optionally recording the visit order."""

    def _make(name: str) -> Callable[[MeshState], MeshState]:
        def node(state: MeshState) -> MeshState:
            if visited is not None:
                visited.append(name)
            return state.with_step(StepResult(worker=name, result=f"{name}-out"))

        return node

    return {name: _make(name) for name in names}


def _recorded(*decisions: list[str], task: str = "t") -> Plan:
    """A plan as the recorder would produce it: one entry per routing decision."""
    steps = iter([*decisions, []])

    def _route(state: MeshState) -> RouteDecision:
        chosen = next(steps, [])
        if not chosen:
            return RouteDecision(next=FINISH)
        if len(chosen) > 1:
            return RouteDecision(next=FINISH, parallel=chosen)
        return RouteDecision(next=chosen[0])

    recorder = PlanRecorder(_route)
    state = MeshState(task=task)
    for _ in range(len(decisions) + 1):
        recorder(state)
    return recorder.plan(task)


class TestRecord:
    def test_records_each_step_in_order(self) -> None:
        plan = _recorded(["a"], ["b"])
        assert [s.workers for s in plan.steps] == [["a"], ["b"]]

    def test_a_parallel_fan_out_is_recovered_from_the_step_index(self) -> None:
        # StepResult.step repeats across workers dispatched together, so grouping by it
        # recovers the shape the supervisor chose — no extra bookkeeping needed.
        plan = _recorded(["a", "b"], ["c"])
        assert [s.workers for s in plan.steps] == [["a", "b"], ["c"]]

    def test_the_task_is_stored_as_a_hash_not_text(self) -> None:
        # A plan lives on disk; the same discipline that keeps raw content out of the
        # audit ledger applies here.
        plan = _recorded(["a"], task="a very secret task")
        assert "secret" not in plan.model_dump_json()
        assert plan.task_sha256 == hash_task("a very secret task")

    def test_a_plan_knows_which_task_it_belongs_to(self) -> None:
        plan = _recorded(["a"])
        assert plan.matches("t") and not plan.matches("something else")

    def test_an_empty_run_records_an_empty_plan(self) -> None:
        assert _recorded().steps == []


class TestReplay:
    def test_replays_the_recorded_sequence(self) -> None:
        plan = Plan(task_sha256=hash_task("t"), steps=[
            PlanStep(step=0, workers=["a"]), PlanStep(step=1, workers=["b"])
        ])
        route = replay_route(plan)
        state = MeshState(task="t")
        assert route(state).next == "a"
        assert route(state).next == "b"

    def test_finishes_when_the_recording_runs_out(self) -> None:
        route = replay_route(Plan(task_sha256=hash_task("t"), steps=[
            PlanStep(step=0, workers=["a"])
        ]))
        state = MeshState(task="t")
        route(state)
        assert route(state).next == FINISH

    def test_a_fan_out_replays_as_parallel(self) -> None:
        route = replay_route(Plan(task_sha256=hash_task("t"), steps=[
            PlanStep(step=0, workers=["a", "b"])
        ]))
        decision: RouteDecision = route(MeshState(task="t"))
        assert decision.parallel == ["a", "b"]

    def test_an_empty_plan_finishes_immediately(self) -> None:
        assert replay_route(Plan(task_sha256=hash_task("t")))(MeshState(task="t")).next == FINISH


class TestZeroSupervisorCalls:
    def test_replay_makes_NO_supervisor_calls(self) -> None:
        """The whole point of the slice, asserted directly.

        A replayed run must cost nothing in routing tokens — that is what makes it usable
        as a regression test and as a debugging tool.
        """
        calls: list[str] = []

        def _supervisor(state: MeshState) -> RouteDecision:
            calls.append("routed")
            return RouteDecision(next=FINISH)

        plan = _recorded(["a"], ["b"])
        LocalEngine().run(
            MeshState(task="t"),
            nodes=_nodes(("a", "b")),
            route=replay_route(plan),
            max_steps=8,
        )
        assert calls == []  # `_supervisor` was never wired in — nothing routed by LLM

    def test_a_replayed_run_visits_the_recorded_workers(self) -> None:
        visited: list[str] = []
        plan = _recorded(["a"], ["b"])
        LocalEngine().run(
            MeshState(task="t"),
            nodes=_nodes(("a", "b"), visited=visited),
            route=replay_route(plan),
            max_steps=8,
        )
        assert visited == ["a", "b"]


class TestDivergence:
    def test_a_missing_worker_fails_closed(self) -> None:
        """A replay that quietly diverges looks like a reproduction and is not."""
        plan = _recorded(["a"], ["gone"])
        with pytest.raises(PlanDivergence, match="gone"):
            replay_route(plan, declared={"a"})

    def test_the_error_names_every_missing_worker(self) -> None:
        plan = _recorded(["x"], ["y"])
        with pytest.raises(PlanDivergence) as exc:
            replay_route(plan, declared=set())
        assert "x" in str(exc.value) and "y" in str(exc.value)

    def test_a_matching_roster_replays_fine(self) -> None:
        plan = _recorded(["a"])
        assert replay_route(plan, declared={"a", "b"}) is not None

    def test_no_roster_supplied_skips_the_check(self) -> None:
        # Callers that cannot know the roster (a raw plan inspection) still work.
        plan = _recorded(["whoever"])
        assert replay_route(plan) is not None


class TestStorage:
    def test_a_plan_round_trips(self, tmp_path: Path) -> None:
        plan = _recorded(["a", "b"], ["c"])
        save_plan(tmp_path, "assistant", "run-1", plan)
        loaded = load_plan(tmp_path, "assistant", "run-1")
        assert [s.workers for s in loaded.steps] == [["a", "b"], ["c"]]

    def test_a_missing_plan_raises(self, tmp_path: Path) -> None:
        with pytest.raises(PlanNotFound):
            load_plan(tmp_path, "assistant", "never-ran")

    def test_plans_are_listed_per_agent(self, tmp_path: Path) -> None:
        plan = _recorded(["a"])
        save_plan(tmp_path, "assistant", "b-run", plan)
        save_plan(tmp_path, "assistant", "a-run", plan)
        save_plan(tmp_path, "other", "z-run", plan)
        assert list_plans(tmp_path, "assistant") == ["a-run", "b-run"]

    def test_listing_an_agent_with_no_plans_is_empty(self, tmp_path: Path) -> None:
        assert list_plans(tmp_path, "nobody") == []

    @pytest.mark.parametrize("bad", ["../../etc", "/etc/passwd", "a/b", ""])
    def test_a_traversing_thread_id_is_refused(self, tmp_path: Path, bad: str) -> None:
        # `Path(base) / "../../etc"` silently escapes — the hole PR #35 caught in distill
        # drafts and S5b caught in sessions. Guarded before any path join.
        with pytest.raises(PlanDivergence):
            plan_path(tmp_path, "assistant", bad)

    def test_a_traversing_agent_name_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(PlanDivergence):
            plan_path(tmp_path, "../../etc", "run-1")
