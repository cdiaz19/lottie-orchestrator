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
    lg_workers = [s.worker for s in lg.state.history]
    loc_workers = [s.worker for s in loc.state.history]
    assert lg_workers == loc_workers == ["a", "b"]
    assert lg.state.final == loc.state.final == "b:done"
    assert lg.thread_id == "t1"


def test_langgraph_engine_lists_checkpoint_history() -> None:
    from lottie.mesh.langgraph_engine import LangGraphEngine

    eng = LangGraphEngine()
    nodes = {"a": _node("a")}
    eng.run(
        MeshState(task="t"),
        nodes=nodes,
        route=_scripted_route(["a", FINISH]),
        max_steps=8,
        thread_id="hist1",
    )
    snapshots = eng.history("hist1", nodes=nodes, route=_scripted_route([]))
    assert len(snapshots) >= 1
    assert any(s.history for s in snapshots)  # some checkpoint captured the worker step


def test_sequential_runs_do_not_contaminate_checkpoint() -> None:
    from lottie.mesh.langgraph_engine import LangGraphEngine

    eng = LangGraphEngine()  # reused across two runs (mimics AgentService's cached instance)
    nodes = {"a": _node("a")}

    r1 = eng.run(
        MeshState(task="t1"), nodes=nodes, route=_scripted_route(["a", FINISH]), max_steps=8
    )
    r2 = eng.run(
        MeshState(task="t2"), nodes=nodes, route=_scripted_route(["a", FINISH]), max_steps=8
    )

    # each run is isolated: run 2 must NOT inherit run 1's history
    assert [s.worker for s in r1.state.history] == ["a"]
    assert [s.worker for s in r2.state.history] == ["a"]
    # distinct auto-generated thread ids
    assert r1.thread_id != r2.thread_id
