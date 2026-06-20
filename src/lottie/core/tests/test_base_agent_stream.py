from __future__ import annotations

from collections.abc import Generator, Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel

from lottie.core.base_agent import BaseAgent, NotStreamable
from lottie.core.metrics import RunContext
from lottie.governance.audit import SqliteAuditLogger
from lottie.governance.cost import BudgetExceeded, CostGate
from lottie.governance.policy import PolicyDenied, PolicyGate
from lottie.governance.schema import AuditRecord
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


def test_run_stream_yields_incrementally(tmp_path: Path) -> None:
    agent = _StreamingAgent(MockLLMProvider(["the launch post"]), audit=SqliteAuditLogger(tmp_path))
    pieces = list(agent.run_stream(_In(q="hi")))
    assert len(pieces) > 1 and "".join(pieces) == "the launch post"
    rows = SqliteAuditLogger(tmp_path).query()
    assert rows[0].status == "ok" and rows[0].root is True


def test_run_stream_policy_deny_blocks_before_any_piece(tmp_path: Path) -> None:
    agent = _StreamingAgent(MockLLMProvider(["x y z"]), audit=SqliteAuditLogger(tmp_path))
    agent.set_policy(PolicyGate(["shell"], allow=set(), deny={"shell"}, escalate=set()))
    gen = agent.run_stream(_In(q="hi"))
    with pytest.raises(PolicyDenied):
        next(gen)
    rows = SqliteAuditLogger(tmp_path).query()
    assert [r.status for r in rows] == ["denied"]


def test_run_stream_over_budget_blocks_pre_run(tmp_path: Path) -> None:
    logger = SqliteAuditLogger(tmp_path)
    agent = _StreamingAgent(MockLLMProvider(["x y z"]), name="digest", audit=logger)
    logger.log(AuditRecord(
        ts="2026-06-20T10:00:00+00:00", agent="digest", provider="fix/model", status="ok",
        root=True, input_sha256="a" * 64, output_sha256="b" * 64, input_tokens=0,
        output_tokens=0, cost_usd=0.10, latency_ms=1.0, error=None,
    ))
    agent.set_cost_gate(CostGate("digest", 0.10, logger))
    gen = agent.run_stream(_In(q="hi"))
    with pytest.raises(BudgetExceeded):
        next(gen)
    assert "budget_exceeded" in [r.status for r in SqliteAuditLogger(tmp_path).query(limit=20)]


def test_run_stream_early_close_audits_partial(tmp_path: Path) -> None:
    agent = _StreamingAgent(MockLLMProvider(["alpha beta gamma"]), audit=SqliteAuditLogger(tmp_path))  # noqa: E501
    gen = agent.run_stream(_In(q="hi"))
    assert next(gen)  # pull the first delta
    gen.close()
    rows = SqliteAuditLogger(tmp_path).query()
    assert rows[0].status == "error" and rows[0].error == "stream closed before completion"


def test_run_stream_usage_parity_with_run(tmp_path: Path) -> None:
    a_run = _StreamingAgent(_UsageProvider(), audit=SqliteAuditLogger(tmp_path / "r"))
    a_run.run(_In(q="hi"))
    a_stream = _StreamingAgent(_UsageProvider(), audit=SqliteAuditLogger(tmp_path / "s"))
    list(a_stream.run_stream(_In(q="hi")))
    rm, sm = a_run.last_metrics, a_stream.last_metrics
    assert rm is not None and sm is not None
    assert (rm.input_tokens, rm.output_tokens, rm.cost_usd) == (11, 7, 0.25)
    assert (sm.input_tokens, sm.output_tokens, sm.cost_usd) == (11, 7, 0.25)
