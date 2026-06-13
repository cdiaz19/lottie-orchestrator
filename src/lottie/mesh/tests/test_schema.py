from __future__ import annotations

from lottie.mesh.schema import (
    FINISH,
    MeshInput,
    MeshOutput,
    MeshState,
    RouteDecision,
    StepResult,
)


def test_mesh_state_defaults_and_add_step() -> None:
    state = MeshState(task="summarize X")
    assert state.history == [] and state.final is None
    nxt = state.with_step(StepResult(worker="research", result="r"))
    assert [s.worker for s in nxt.history] == ["research"]
    # original is not mutated (with_step returns a new state)
    assert state.history == []


def test_route_decision_and_finish_sentinel() -> None:
    assert FINISH == "FINISH"
    assert RouteDecision(next="research").next == "research"
    assert RouteDecision(next=FINISH).next == "FINISH"


def test_mesh_io_models() -> None:
    assert MeshInput(task="t").max_steps == 8
    out = MeshOutput(final="done", history=[StepResult(worker="critic", result="ok")])
    assert out.final == "done" and out.history[0].worker == "critic"


def test_mesh_errors_hierarchy() -> None:
    from lottie.mesh.errors import (
        CapabilityViolation,
        MeshError,
        MeshStepLimitExceeded,
    )

    assert issubclass(CapabilityViolation, MeshError)
    assert issubclass(MeshStepLimitExceeded, MeshError)


def test_agent_config_has_workers_field() -> None:
    from lottie.project.config import AgentConfig

    cfg = AgentConfig(provider="mock/x", workers=["research", "critic"])
    assert cfg.workers == ["research", "critic"]
    # default empty for non-mesh agents
    assert AgentConfig(provider="mock/x").workers == []


def test_package_exports() -> None:
    import lottie.mesh as m

    for sym in ("MeshAgent", "MeshInput", "MeshOutput", "MeshState", "LocalEngine"):
        assert hasattr(m, sym), sym


def test_step_result_has_step_index_default_zero() -> None:
    from lottie.mesh.schema import StepResult

    assert StepResult(worker="w", result="r").step == 0
    assert StepResult(worker="w", result="r", step=3).step == 3


def test_mesh_state_history_still_appends_via_with_step() -> None:
    from lottie.mesh.schema import MeshState, StepResult

    s = MeshState(task="t").with_step(StepResult(worker="a", result="x"))
    assert [h.worker for h in s.history] == ["a"]
