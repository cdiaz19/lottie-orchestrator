"""Agentic-loop hygiene: max_turns (LLM-completion cap) + the _verify post-condition hook."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from pydantic import BaseModel

from lottie.core.base_agent import BaseAgent, TurnLimitExceeded
from lottie.llm import LLMResponse, Message, MockLLMProvider
from lottie.llm.base import LLMProvider, TokenUsage


class _In(BaseModel):
    q: str


class _Out(BaseModel):
    a: str


class _Provider(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def model(self) -> str:
        return "sim"

    def complete(
        self, messages: list[Message], model_params: Mapping[str, object] | None = None
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content="ok", usage=TokenUsage(), model="sim", cost_usd=0.0)


class _NCallAgent(BaseAgent[_In, _Out]):
    def __init__(self, provider: LLMProvider, n: int) -> None:
        super().__init__(provider)
        self._n = n

    def _execute(self, data: _In) -> _Out:
        for _ in range(self._n):
            self.complete([Message(role="user", content="x")])
        return _Out(a="done")


class TestMaxTurns:
    def test_exceeding_max_turns_raises(self) -> None:
        p = _Provider()
        agent = _NCallAgent(p, n=3)
        agent.set_run_limits(max_turns=2)
        with pytest.raises(TurnLimitExceeded):
            agent.run(_In(q="hi"))
        assert p.calls == 3  # raised on the 3rd (which is > 2), before a 4th

    def test_under_cap_runs(self) -> None:
        agent = _NCallAgent(_Provider(), n=2)
        agent.set_run_limits(max_turns=5)
        assert agent.run(_In(q="hi")).a == "done"

    def test_none_is_unlimited(self) -> None:
        agent = _NCallAgent(_Provider(), n=10)  # default max_turns None
        assert agent.run(_In(q="hi")).a == "done"


class _VerifyAgent(BaseAgent[_In, _Out]):
    def __init__(self, provider: LLMProvider, *, reject: bool) -> None:
        super().__init__(provider)
        self._reject = reject
        self.verified_with: str | None = None

    def _execute(self, data: _In) -> _Out:
        return _Out(a=f"out:{data.q}")

    def _verify(self, data: _In, output: _Out) -> None:
        self.verified_with = output.a
        if self._reject:
            raise ValueError("post-condition failed")


class TestVerifyHook:
    def test_verify_runs_after_execute(self) -> None:
        agent = _VerifyAgent(MockLLMProvider(["x"]), reject=False)
        out = agent.run(_In(q="hi"))
        assert out.a == "out:hi"
        assert agent.verified_with == "out:hi"  # called with the produced output

    def test_verify_failure_fails_the_run(self) -> None:
        agent = _VerifyAgent(MockLLMProvider(["x"]), reject=True)
        with pytest.raises(ValueError, match="post-condition failed"):
            agent.run(_In(q="hi"))

    def test_default_verify_is_noop(self) -> None:
        agent = _NCallAgent(_Provider(), n=1)  # does not override _verify
        assert agent.run(_In(q="hi")).a == "done"
