"""Per-run token cap + reservation settle lifecycle at the BaseAgent chokepoint."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import BaseModel

from lottie.core.base_agent import BaseAgent
from lottie.governance.audit import SqliteAuditLogger
from lottie.governance.cost import BudgetExceeded, CostGate, TokenCapExceeded
from lottie.llm import LLMResponse, Message, MockLLMProvider
from lottie.llm.base import LLMProvider, TokenUsage


class _In(BaseModel):
    q: str


class _Out(BaseModel):
    a: str


class _UsageProvider(LLMProvider):
    """Reports fixed non-zero usage per complete() call."""

    def __init__(self, in_tok: int, out_tok: int) -> None:
        self._in, self._out = in_tok, out_tok
        self.calls = 0

    @property
    def model(self) -> str:
        return "usage/sim"

    def complete(
        self, messages: list[Message], model_params: Mapping[str, object] | None = None
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            content="ok",
            usage=TokenUsage(input_tokens=self._in, output_tokens=self._out),
            model="usage/sim",
            cost_usd=0.0,
        )


class _MultiCallAgent(BaseAgent[_In, _Out]):
    """Calls complete() twice (a two-'turn' run)."""

    def _execute(self, data: _In) -> _Out:
        self.complete([Message(role="user", content="1")])
        self.complete([Message(role="user", content="2")])
        return _Out(a="done")


class TestTokenCap:
    def test_cap_aborts_runaway_run(self) -> None:
        provider = _UsageProvider(3, 3)  # 6 tokens per call
        agent = _MultiCallAgent(provider)
        agent.set_run_limits(max_run_tokens=5)  # first call (6) already exceeds
        with pytest.raises(TokenCapExceeded):
            agent.run(_In(q="hi"))
        assert provider.calls == 1  # aborted before the second call

    def test_under_cap_runs_clean(self) -> None:
        provider = _UsageProvider(1, 1)  # 2 per call, 4 total
        agent = _MultiCallAgent(provider)
        agent.set_run_limits(max_run_tokens=100)
        assert agent.run(_In(q="hi")).a == "done"
        assert provider.calls == 2

    def test_none_is_unlimited(self) -> None:
        agent = _MultiCallAgent(_UsageProvider(1000, 1000))
        # default max_run_tokens is None
        assert agent.run(_In(q="hi")).a == "done"


class _Spy(BaseAgent[_In, _Out]):
    def __init__(self, llm: object, audit: SqliteAuditLogger, *, boom: bool = False) -> None:
        super().__init__(llm, audit=audit)  # type: ignore[arg-type]
        self._boom = boom

    def _execute(self, data: _In) -> _Out:
        if self._boom:
            raise ValueError("boom")
        return _Out(a="ok")


def _reserved_rows(tmp_path: Path, agent: str) -> int:
    conn = SqliteAuditLogger(tmp_path)._connect()
    return int(
        conn.execute(
            "SELECT COUNT(*) FROM reservations WHERE agent = ?", (agent,)
        ).fetchone()[0]
    )


class TestReservationSettle:
    def test_reservation_settled_on_success(self, tmp_path: Path) -> None:
        lg = SqliteAuditLogger(tmp_path)
        agent = _Spy(MockLLMProvider(["x"]), lg)
        agent.set_cost_gate(CostGate("_Spy", 10.0, lg, max_run_usd=2.0))
        agent.run(_In(q="hi"))
        assert _reserved_rows(tmp_path, "_Spy") == 0  # released after the run

    def test_reservation_settled_on_failure(self, tmp_path: Path) -> None:
        lg = SqliteAuditLogger(tmp_path)
        agent = _Spy(MockLLMProvider(["x"]), lg, boom=True)
        agent.set_cost_gate(CostGate("_Spy", 10.0, lg, max_run_usd=2.0))
        with pytest.raises(ValueError, match="boom"):
            agent.run(_In(q="hi"))
        assert _reserved_rows(tmp_path, "_Spy") == 0  # released even on crash — no leak

    def test_over_reserved_run_blocked_and_audited(self, tmp_path: Path) -> None:
        lg = SqliteAuditLogger(tmp_path)
        agent = _Spy(MockLLMProvider(["x"]), lg)
        # budget 5, per-run ceiling 6 -> first reservation already exceeds
        agent.set_cost_gate(CostGate("_Spy", 5.0, lg, max_run_usd=6.0))
        with pytest.raises(BudgetExceeded):
            agent.run(_In(q="hi"))
        rows = SqliteAuditLogger(tmp_path).query(agent="_Spy")
        assert rows and rows[0].status == "budget_exceeded"
