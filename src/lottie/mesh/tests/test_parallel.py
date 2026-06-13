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
