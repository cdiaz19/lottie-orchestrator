from datetime import UTC, datetime

from pydantic import BaseModel

from lottie.core import BaseAgent
from lottie.core.metrics import RunMetrics
from lottie.llm import MockLLMProvider
from lottie.memory.mock import MockMemoryClient
from lottie.memory.schema import MemoryOrigin, MemoryQuery, MemoryTier


class _In(BaseModel):
    q: str


class _Out(BaseModel):
    a: str


class _Answer(BaseAgent[_In, _Out]):
    def _execute(self, data: _In) -> _Out:
        return _Out(a="answer")


def test_reflection_disabled_writes_nothing() -> None:
    mem = MockMemoryClient()
    agent = _Answer(llm=MockLLMProvider(["lesson one"]), memory=mem)
    agent.run(_In(q="hi"))
    assert mem.records == []


def test_reflection_enabled_writes_lessons_with_provenance() -> None:
    mem = MockMemoryClient()
    # first canned response = _execute has no LLM call here, so the ONLY completion is reflection
    agent = _Answer(
        llm=MockLLMProvider(["always validate units\ncache the parsed config"]), memory=mem
    )
    agent.set_reflect(enabled=True, namespace="ns")
    agent.run(_In(q="hi"))
    notes = mem.recall(MemoryQuery(text="", namespace="ns", tier=MemoryTier.SEMANTIC)).hits
    contents = {h.record.content for h in notes}
    assert "always validate units" in contents
    assert "cache the parsed config" in contents
    assert all(h.record.origin is MemoryOrigin.REFLECTION for h in notes)
    assert all(h.record.source_agent == agent.name for h in notes)


def test_reflection_failure_does_not_break_run() -> None:
    # NullMemoryClient (default) raises on write; run must still return normally.
    agent = _Answer(llm=MockLLMProvider(["a lesson"]))
    agent.set_reflect(enabled=True, namespace="ns")
    out = agent.run(_In(q="hi"))
    assert out.a == "answer"


def test_reflection_skipped_when_token_cap_reached() -> None:
    mem = MockMemoryClient()
    agent = _Answer(llm=MockLLMProvider(["should not be written"]), memory=mem)
    agent.set_reflect(enabled=True, namespace="ns")
    # simulate a run that already consumed its token budget
    agent.set_run_limits(max_run_tokens=1)
    agent.last_metrics = RunMetrics(
        name=agent.name, kind="agent", provider=None, timestamp=datetime.now(UTC),
        latency_ms=1.0, input_tokens=5, output_tokens=5, cost_usd=0.0, retry_count=0,
        success=True, version=None, error=None,
    )
    agent._maybe_reflect(_In(q="hi"), _Out(a="answer"))
    assert mem.records == []  # cap already reached -> reflection skipped
