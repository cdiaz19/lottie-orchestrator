from collections.abc import Mapping

import pytest
from pydantic import BaseModel

from lottie.core import BaseAgent
from lottie.llm import LLMProvider, LLMResponse, Message, TokenUsage
from lottie.memory import MemoryNotConfiguredError, MockMemoryClient
from lottie.memory.base import NullMemoryClient
from lottie.memory.schema import MemoryQuery, MemoryRecord


class _In(BaseModel):
    x: int


class _Out(BaseModel):
    text: str


class FakeProvider(LLMProvider):
    def __init__(self, model: str = "fake/m") -> None:
        self._model = model
        self.calls = 0

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            content="ok",
            usage=TokenUsage(input_tokens=7, output_tokens=3),
            model=self._model,
            cost_usd=0.05,
        )


class EchoAgent(BaseAgent[_In, _Out]):
    def _execute(self, data: _In) -> _Out:
        resp = self.complete([Message(role="user", content=str(data.x))])
        return _Out(text=resp.content)


class TwoCallAgent(BaseAgent[_In, _Out]):
    def _execute(self, data: _In) -> _Out:
        self.complete([Message(role="user", content="a")])
        resp = self.complete([Message(role="user", content="b")])
        return _Out(text=resp.content)


class _MemAgent(BaseAgent[_In, _Out]):
    def _execute(self, data: _In) -> _Out:
        return _Out(text=str(data.x))


def test_agent_returns_output() -> None:
    agent = EchoAgent(FakeProvider(), enable_benchmarks=False)
    assert agent.run(_In(x=1)).text == "ok"


def test_agent_provider_is_model() -> None:
    agent = EchoAgent(FakeProvider("anthropic/claude-sonnet-4-6"), enable_benchmarks=False)
    agent.run(_In(x=1))
    assert agent.provider == "anthropic/claude-sonnet-4-6"
    assert agent.last_metrics is not None
    assert agent.last_metrics.provider == "anthropic/claude-sonnet-4-6"
    assert agent.last_metrics.kind == "agent"


def test_agent_accumulates_tokens_and_cost() -> None:
    llm = FakeProvider()
    agent = EchoAgent(llm, enable_benchmarks=False)
    agent.run(_In(x=1))
    m = agent.last_metrics
    assert m is not None
    assert m.input_tokens == 7
    assert m.output_tokens == 3
    assert round(m.cost_usd, 4) == 0.05
    assert llm.calls == 1


def test_agent_accumulates_multiple_calls() -> None:
    agent = TwoCallAgent(FakeProvider(), enable_benchmarks=False)
    agent.run(_In(x=1))
    m = agent.last_metrics
    assert m is not None
    assert m.input_tokens == 14
    assert m.output_tokens == 6
    assert round(m.cost_usd, 4) == 0.10


def test_complete_outside_run_returns_response() -> None:
    agent = EchoAgent(FakeProvider(), enable_benchmarks=False)
    resp = agent.complete([Message(role="user", content="hi")])
    assert resp.content == "ok"


def test_base_agent_default_memory_is_null() -> None:
    agent = _MemAgent(FakeProvider())
    assert isinstance(agent.memory, NullMemoryClient)
    with pytest.raises(MemoryNotConfiguredError):
        agent.memory.recall(MemoryQuery(text="", namespace="demo"))


def test_base_agent_uses_injected_memory() -> None:
    client = MockMemoryClient()
    agent = _MemAgent(FakeProvider(), memory=client)
    assert agent.memory is client
    agent.memory.remember(MemoryRecord(content="hi", namespace="demo"))
    assert client.records[0].content == "hi"
