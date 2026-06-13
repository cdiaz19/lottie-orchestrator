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
