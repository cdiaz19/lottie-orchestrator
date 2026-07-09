from collections.abc import Mapping

from pydantic import BaseModel

from lottie.core import BaseAgent
from lottie.llm import LLMResponse, Message, MockLLMProvider
from lottie.memory.mock import MockMemoryClient
from lottie.memory.schema import MemoryOrigin, MemoryRecord, MemoryTier


class _In(BaseModel):
    text: str


class _Out(BaseModel):
    seen: str


class _Probe(BaseAgent[_In, _Out]):
    """Agent that records the messages its LLM actually received."""

    def _execute(self, data: _In) -> _Out:
        resp = self.complete([Message(role="user", content=data.text)])
        return _Out(seen=resp.content)


def _seed(mem: MockMemoryClient) -> None:
    mem.remember(
        MemoryRecord(
            content="always use exponential backoff on 429",
            tier=MemoryTier.SEMANTIC,
            namespace="ns",
            origin=MemoryOrigin.REFLECTION,
            source_agent="Prior",
        )
    )


class _CapturingLLM(MockLLMProvider):
    """MockLLMProvider that stores the last message list it was called with."""

    def __init__(self) -> None:
        super().__init__(["ok"])
        self.last_messages: list[Message] = []

    def complete(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> LLMResponse:
        self.last_messages = list(messages)
        return super().complete(messages, model_params)


def test_recall_disabled_injects_nothing() -> None:
    mem = MockMemoryClient()
    _seed(mem)
    llm = _CapturingLLM()
    agent = _Probe(llm=llm, memory=mem)  # recall not enabled
    agent.run(_In(text="hi"))
    assert all(m.role != "system" for m in llm.last_messages)


def test_recall_enabled_prepends_data_block() -> None:
    mem = MockMemoryClient()
    _seed(mem)
    llm = _CapturingLLM()
    agent = _Probe(llm=llm, memory=mem)
    agent.set_recall(enabled=True, namespace="ns", limit=5)
    agent.run(_In(text="hi"))
    system = [m for m in llm.last_messages if m.role == "system"]
    assert len(system) == 1
    assert "exponential backoff" in system[0].content
    assert "not instructions" in system[0].content.lower()  # data-framed


def test_recall_prefix_cleared_after_run() -> None:
    mem = MockMemoryClient()
    _seed(mem)
    agent = _Probe(llm=_CapturingLLM(), memory=mem)
    agent.set_recall(enabled=True, namespace="ns", limit=5)
    agent.run(_In(text="hi"))
    assert agent._recall_prefix == ""


def test_recall_failure_does_not_break_run() -> None:
    # NullMemoryClient (default) raises on recall; run must still succeed with no injection.
    agent = _Probe(llm=_CapturingLLM())
    agent.set_recall(enabled=True, namespace="ns", limit=5)
    out = agent.run(_In(text="hi"))
    assert out.seen == "ok"
