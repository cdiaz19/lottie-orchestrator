"""A mesh run records its own plan (E6), end to end."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path

from lottie.llm import MockLLMProvider
from lottie.mesh.base import MeshAgent
from lottie.mesh.plan import PlanRecorder, list_plans, load_plan, replay_route
from lottie.mesh.schema import MeshInput, MeshState, StepResult


def _nodes() -> Mapping[str, object]:
    def _make(name: str):  # type: ignore[no-untyped-def]
        def node(state: MeshState) -> MeshState:
            return state.with_step(StepResult(worker=name, result=f"{name}-out"))

        return node

    return {"alpha": _make("alpha"), "beta": _make("beta")}


def _mesh(root: Path | None, responses: list[str]) -> MeshAgent:
    agent = MeshAgent(
        MockLLMProvider(responses),
        nodes=_nodes(),  # type: ignore[arg-type]
        descriptions={"alpha": "does alpha", "beta": "does beta"},
        enable_benchmarks=False,
    )
    agent.set_plans_root(root)
    return agent


def test_no_root_records_nothing(tmp_path: Path) -> None:
    # A directly-constructed mesh (tests, library use) leaves no artefacts.
    _mesh(None, ["alpha", "FINISH"]).run(MeshInput(task="t"))
    assert list_plans(tmp_path, "MeshAgent") == []


def test_a_completed_run_records_its_plan(tmp_path: Path) -> None:
    _mesh(tmp_path, ["alpha", "beta", "FINISH"]).run(MeshInput(task="t"))
    threads = list_plans(tmp_path, "MeshAgent")
    assert len(threads) == 1


def test_the_recorded_plan_matches_what_ran(tmp_path: Path) -> None:
    _mesh(tmp_path, ["alpha", "beta", "FINISH"]).run(MeshInput(task="t"))
    thread = list_plans(tmp_path, "MeshAgent")[0]
    plan = load_plan(tmp_path, "MeshAgent", thread)
    assert [s.workers for s in plan.steps] == [["alpha"], ["beta"]]


def test_the_plan_binds_to_its_task(tmp_path: Path) -> None:
    _mesh(tmp_path, ["alpha", "FINISH"]).run(MeshInput(task="the real task"))
    thread = list_plans(tmp_path, "MeshAgent")[0]
    plan = load_plan(tmp_path, "MeshAgent", thread)
    assert plan.matches("the real task") and not plan.matches("a different task")


def test_the_task_text_is_not_written_to_disk(tmp_path: Path) -> None:
    _mesh(tmp_path, ["alpha", "FINISH"]).run(MeshInput(task="SENSITIVE_TASK_TEXT"))
    thread = list_plans(tmp_path, "MeshAgent")[0]
    from lottie.mesh.plan import plan_path

    assert "SENSITIVE_TASK_TEXT" not in plan_path(tmp_path, "MeshAgent", thread).read_text()


def test_a_recording_failure_never_fails_the_run(tmp_path: Path) -> None:
    agent = _mesh(tmp_path, ["alpha", "FINISH"])
    # An unusable root: recording cannot work, but the run already succeeded.
    agent.set_plans_root(tmp_path / "nested" / "\0bad")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = agent.run(MeshInput(task="t"))
    assert out.status == "complete"


def test_a_parallel_fan_out_is_recorded_as_one_step(tmp_path: Path) -> None:
    """The case that broke the first implementation.

    Inferring the plan from `MeshState.history` grouped by `StepResult.step` looked
    cheaper, but only LangGraphEngine populates `step` — LocalEngine leaves it at 0, so
    every sequential step collapsed into one phantom fan-out. Recording at the routing
    seam is exact regardless of engine.
    """
    from lottie.mesh.schema import FINISH, RouteDecision

    decisions = iter([RouteDecision(next=FINISH, parallel=["alpha", "beta"]),
                      RouteDecision(next=FINISH)])
    recorder = PlanRecorder(lambda state: next(decisions))
    state = MeshState(task="t")
    recorder(state)
    recorder(state)
    assert [s.workers for s in recorder.plan("t").steps] == [["alpha", "beta"]]


def test_sequential_steps_are_recorded_separately(tmp_path: Path) -> None:
    from lottie.mesh.schema import FINISH, RouteDecision

    decisions = iter([RouteDecision(next="alpha"), RouteDecision(next="beta"),
                      RouteDecision(next=FINISH)])
    recorder = PlanRecorder(lambda state: next(decisions))
    state = MeshState(task="t")
    for _ in range(3):
        recorder(state)
    assert [s.workers for s in recorder.plan("t").steps] == [["alpha"], ["beta"]]


def test_a_replayed_run_needs_no_supervisor_responses(tmp_path: Path) -> None:
    """The payoff: a recorded flow re-runs with ZERO routing calls.

    The mock is given NO responses at all — any supervisor call would raise.
    """
    _mesh(tmp_path, ["alpha", "beta", "FINISH"]).run(MeshInput(task="t"))
    thread = list_plans(tmp_path, "MeshAgent")[0]
    plan = load_plan(tmp_path, "MeshAgent", thread)

    from lottie.mesh.local import LocalEngine

    result = LocalEngine().run(
        MeshState(task="t"),
        nodes=_nodes(),  # type: ignore[arg-type]
        route=replay_route(plan, declared={"alpha", "beta"}),
        max_steps=8,
    )
    assert [s.worker for s in result.state.history] == ["alpha", "beta"]
