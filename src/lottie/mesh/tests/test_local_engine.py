from __future__ import annotations

from collections.abc import Callable

import pytest

from lottie.mesh.errors import MeshStepLimitExceeded
from lottie.mesh.local import LocalEngine
from lottie.mesh.schema import FINISH, MeshState, RouteDecision, StepResult


def _node(name: str) -> Callable[[MeshState], MeshState]:
    def _run(state: MeshState) -> MeshState:
        return state.with_step(StepResult(worker=name, result=f"{name}:{state.task}"))

    return _run


def test_engine_runs_route_until_finish() -> None:
    nodes = {"a": _node("a"), "b": _node("b")}
    script = iter(["a", "b", FINISH])

    def route(state: MeshState) -> RouteDecision:
        return RouteDecision(next=next(script))

    final = LocalEngine().run(MeshState(task="t"), nodes=nodes, route=route, max_steps=8)
    assert [s.worker for s in final.history] == ["a", "b"]
    # final is set to the last step's result
    assert final.final == "b:t"


def test_engine_finish_with_no_steps_sets_empty_final() -> None:
    def route(state: MeshState) -> RouteDecision:
        return RouteDecision(next=FINISH)

    final = LocalEngine().run(MeshState(task="t"), nodes={}, route=route, max_steps=8)
    assert final.final == "" and final.history == []


def test_engine_raises_on_step_limit() -> None:
    nodes = {"a": _node("a")}

    def route(state: MeshState) -> RouteDecision:
        return RouteDecision(next="a")  # never FINISH

    with pytest.raises(MeshStepLimitExceeded):
        LocalEngine().run(MeshState(task="t"), nodes=nodes, route=route, max_steps=3)
