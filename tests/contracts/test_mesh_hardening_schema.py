"""Round-trip contract tests for Phase-3 mesh-hardening schemas (CLAUDE.md rule 2)."""

from __future__ import annotations

from typing import Literal

from lottie.mesh.schema import (
    ApprovalDecision,
    MeshOutput,
    MeshRunResult,
    MeshState,
    PendingApproval,
    RouteDecision,
    StepResult,
)


def test_mesh_run_result_roundtrip() -> None:
    r = MeshRunResult(
        state=MeshState(task="t", history=[StepResult(worker="a", result="x", step=0)]),
        status="interrupted",
        thread_id="th1",
        pending=PendingApproval(worker="deploy", proposed_input={"task": "t"}),
    )
    assert MeshRunResult.model_validate_json(r.model_dump_json()) == r


def test_pending_approval_roundtrip() -> None:
    p = PendingApproval(worker="deploy", proposed_input={"k": "v"})
    assert PendingApproval.model_validate_json(p.model_dump_json()) == p


def test_approval_decision_roundtrip() -> None:
    actions: list[Literal["approve", "reject"]] = ["approve", "reject"]
    for action in actions:
        d = ApprovalDecision(action=action, edited_input={"task": "edited"})
        assert ApprovalDecision.model_validate_json(d.model_dump_json()) == d


def test_mesh_output_interrupted_roundtrip() -> None:
    o = MeshOutput(
        final="",
        history=[StepResult(worker="a", result="x", step=1)],
        status="interrupted",
        thread_id="th2",
        pending=PendingApproval(worker="publisher"),
    )
    assert MeshOutput.model_validate_json(o.model_dump_json()) == o


def test_route_decision_parallel_roundtrip() -> None:
    d = RouteDecision(next="FINISH", parallel=["a", "b"])
    assert RouteDecision.model_validate_json(d.model_dump_json()) == d


def test_step_result_step_roundtrip() -> None:
    s = StepResult(worker="a", result="x", step=3, metadata={"k": "v"})
    assert StepResult.model_validate_json(s.model_dump_json()) == s
