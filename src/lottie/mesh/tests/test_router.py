from __future__ import annotations

import pytest

from lottie.llm import LLMResponse, Message, TokenUsage
from lottie.mesh.errors import CapabilityViolation
from lottie.mesh.router import CompleteFn, SupervisorRouter
from lottie.mesh.schema import MeshState


def _complete_returning(text: str) -> CompleteFn:
    def _complete(messages: list[Message]) -> LLMResponse:
        return LLMResponse(content=text, usage=TokenUsage(), model="mock/mock-model")

    return _complete


_WORKERS = {"research": "Finds and summarizes knowledge.", "critic": "Reviews a draft."}


def test_router_returns_validated_worker() -> None:
    router = SupervisorRouter(_complete_returning("research"))
    decision = router.route(MeshState(task="t"), _WORKERS)
    assert decision.next == "research"


def test_router_accepts_finish() -> None:
    router = SupervisorRouter(_complete_returning("FINISH"))
    assert router.route(MeshState(task="t"), _WORKERS).next == "FINISH"


def test_router_is_case_insensitive_and_trims() -> None:
    router = SupervisorRouter(_complete_returning("  Critic \n"))
    assert router.route(MeshState(task="t"), _WORKERS).next == "critic"


def test_router_rejects_undeclared_worker() -> None:
    router = SupervisorRouter(_complete_returning("hacker"))
    with pytest.raises(CapabilityViolation):
        router.route(MeshState(task="t"), _WORKERS)


def test_router_parses_parallel_fanout() -> None:
    router = SupervisorRouter(_complete_returning("research, critic"))
    decision = router.route(MeshState(task="t"), _WORKERS)
    assert sorted(decision.parallel) == ["critic", "research"]


def test_router_parallel_rejects_undeclared() -> None:
    router = SupervisorRouter(_complete_returning("research, hacker"))
    with pytest.raises(CapabilityViolation):
        router.route(MeshState(task="t"), _WORKERS)
