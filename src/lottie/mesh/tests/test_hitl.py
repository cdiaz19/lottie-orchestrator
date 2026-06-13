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


def _route_then_finish(worker: str) -> Callable[[MeshState], RouteDecision]:
    def route(state: MeshState) -> RouteDecision:
        return RouteDecision(next=worker) if not state.history else RouteDecision(next=FINISH)
    return route


def test_interrupt_before_pauses_then_resume_approve_completes() -> None:
    from lottie.mesh.langgraph_engine import LangGraphEngine

    nodes = {"deploy": _node("deploy")}
    eng = LangGraphEngine(interrupt_before=["deploy"])
    route = _route_then_finish("deploy")

    paused = eng.run(
        MeshState(task="ship it"), nodes=nodes, route=route, max_steps=8, thread_id="h1"
    )
    assert paused.status == "interrupted"
    assert paused.pending is not None and paused.pending.worker == "deploy"

    done = eng.resume("h1", nodes=nodes, route=route, decision=ApprovalDecision(action="approve"))
    assert done.status == "complete"
    assert [s.worker for s in done.state.history] == ["deploy"]


def test_resume_reject_skips_worker() -> None:
    from lottie.mesh.langgraph_engine import LangGraphEngine

    nodes = {"deploy": _node("deploy")}
    eng = LangGraphEngine(interrupt_before=["deploy"])
    route = _route_then_finish("deploy")

    eng.run(MeshState(task="ship it"), nodes=nodes, route=route, max_steps=8, thread_id="h2")
    done = eng.resume("h2", nodes=nodes, route=route, decision=ApprovalDecision(action="reject"))
    assert done.status == "complete"
    assert any(s.worker == "deploy" and "reject" in s.result.lower() for s in done.state.history)
