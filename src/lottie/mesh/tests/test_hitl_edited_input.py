"""HITL edited_input-on-approve: the human-edited MeshState fields reach the resumed worker;
non-editable / invalid edits are rejected fail-closed."""

from __future__ import annotations

import pytest

try:
    import langgraph  # noqa: F401

    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False

pytestmark = pytest.mark.skipif(not _HAS_LANGGRAPH, reason="needs [mesh] extra")

from collections.abc import Callable  # noqa: E402

from lottie.mesh.errors import EditedInputError  # noqa: E402
from lottie.mesh.schema import (  # noqa: E402
    FINISH,
    ApprovalDecision,
    MeshState,
    RouteDecision,
    StepResult,
)


def _echo_task_node(name: str) -> Callable[[MeshState], MeshState]:
    def _run(state: MeshState) -> MeshState:
        return state.with_step(StepResult(worker=name, result=f"{name}:{state.task}"))

    return _run


def _route_then_finish(worker: str) -> Callable[[MeshState], RouteDecision]:
    def route(state: MeshState) -> RouteDecision:
        return RouteDecision(next=worker) if not state.history else RouteDecision(next=FINISH)

    return route


def _paused_engine(thread: str):  # type: ignore[no-untyped-def]
    from lottie.mesh.langgraph_engine import LangGraphEngine

    nodes = {"deploy": _echo_task_node("deploy")}
    route = _route_then_finish("deploy")
    eng = LangGraphEngine(interrupt_before=["deploy"])
    eng.run(MeshState(task="original"), nodes=nodes, route=route, max_steps=8, thread_id=thread)
    return eng, nodes, route


def test_approve_with_edited_input_reaches_worker() -> None:
    eng, nodes, route = _paused_engine("ed1")
    done = eng.resume(
        "ed1", nodes=nodes, route=route,
        decision=ApprovalDecision(action="approve", edited_input={"task": "EDITED"}),
    )
    assert done.status == "complete"
    assert done.state.history[0].result == "deploy:EDITED"  # worker ran on the edited task


def test_approve_with_empty_edit_is_unchanged() -> None:
    eng, nodes, route = _paused_engine("ed2")
    done = eng.resume(
        "ed2", nodes=nodes, route=route,
        decision=ApprovalDecision(action="approve", edited_input={}),
    )
    assert done.state.history[0].result == "deploy:original"


def test_edit_of_non_editable_field_is_rejected() -> None:
    eng, nodes, route = _paused_engine("ed3")
    with pytest.raises(EditedInputError):
        eng.resume(
            "ed3", nodes=nodes, route=route,
            decision=ApprovalDecision(action="approve", edited_input={"history": "x"}),
        )


def test_edit_of_unknown_field_is_rejected() -> None:
    eng, nodes, route = _paused_engine("ed4")
    with pytest.raises(EditedInputError):
        eng.resume(
            "ed4", nodes=nodes, route=route,
            decision=ApprovalDecision(action="approve", edited_input={"bogus": "x"}),
        )


def test_edit_of_final_field_allowed() -> None:
    eng, nodes, route = _paused_engine("ed5")
    done = eng.resume(
        "ed5", nodes=nodes, route=route,
        decision=ApprovalDecision(action="approve", edited_input={"final": "forced"}),
    )
    assert done.status == "complete"  # `final` is an editable string field
