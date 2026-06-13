"""Round-trip contract tests for mesh schemas (no raw dict/str crosses a boundary)."""

from __future__ import annotations

from lottie.mesh.schema import (
    MeshInput,
    MeshOutput,
    MeshState,
    RouteDecision,
    StepResult,
)


def test_mesh_state_roundtrip() -> None:
    state = MeshState(
        task="t", history=[StepResult(worker="research", result="r")], final="f"
    )
    assert MeshState.model_validate_json(state.model_dump_json()) == state


def test_route_decision_roundtrip() -> None:
    d = RouteDecision(next="critic")
    assert RouteDecision.model_validate_json(d.model_dump_json()) == d


def test_mesh_io_roundtrip() -> None:
    mi = MeshInput(task="t", max_steps=4)
    assert MeshInput.model_validate_json(mi.model_dump_json()) == mi
    mo = MeshOutput(final="f", history=[StepResult(worker="critic", result="ok")])
    assert MeshOutput.model_validate_json(mo.model_dump_json()) == mo
