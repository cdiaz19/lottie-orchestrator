from __future__ import annotations

from collections.abc import Generator, Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel

from lottie.core.base_agent import BaseAgent, NotStreamable
from lottie.core.metrics import RunContext
from lottie.governance.audit import SqliteAuditLogger
from lottie.llm import MockLLMProvider
from lottie.llm.base import LLMProvider, LLMResponse, Message, StreamResult, TokenUsage


class _In(BaseModel):
    q: str


class _Out(BaseModel):
    a: str


class _Plain(BaseAgent[_In, _Out]):
    def _execute(self, data: _In) -> _Out:
        return _Out(a=data.q)


class _StreamingAgent(BaseAgent[_In, _Out]):
    def _execute(self, data: _In) -> _Out:
        return _Out(a=self.complete([Message(role="user", content=data.q)]).content)

    def _stream(self, data: _In) -> Iterator[str]:
        yield from self.stream_complete([Message(role="user", content=data.q)])


class _UsageProvider(LLMProvider):
    @property
    def model(self) -> str:
        return "fix/model"

    def complete(self, messages: list[Message], model_params: object = None) -> LLMResponse:
        return LLMResponse(content="the launch post",
                           usage=TokenUsage(input_tokens=11, output_tokens=7),
                           model="fix/model", cost_usd=0.25)

    def stream_complete(
        self,
        messages: list[Message],
        model_params: object = None,
    ) -> Generator[str, None, StreamResult]:
        import re
        yield from re.findall(r"\S+\s*|\s+", "the launch post")
        return StreamResult(usage=TokenUsage(input_tokens=11, output_tokens=7), cost_usd=0.25)


def test_supports_streaming_reflects_override() -> None:
    assert _StreamingAgent.supports_streaming() is True
    assert _Plain.supports_streaming() is False


def test_default_stream_raises_not_streamable(tmp_path: Path) -> None:
    agent = _Plain(MockLLMProvider(["x"]), audit=SqliteAuditLogger(tmp_path))
    with pytest.raises(NotStreamable):
        agent._stream(_In(q="hi"))


def test_stream_complete_accumulates_usage_into_active_ctx(tmp_path: Path) -> None:
    agent = _StreamingAgent(_UsageProvider(), audit=SqliteAuditLogger(tmp_path))
    agent._active_ctx = RunContext()
    deltas = list(agent.stream_complete([Message(role="user", content="hi")]))
    assert "".join(deltas) == "the launch post"
    assert agent._active_ctx.input_tokens == 11
    assert agent._active_ctx.output_tokens == 7
    assert agent._active_ctx.cost_usd == 0.25
