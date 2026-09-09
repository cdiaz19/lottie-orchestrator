"""A run's real cost must reach the audit ledger — the cumulative breaker sums it.

Regression for a silent governance hole: `Pipeline` publishes the run's usage
accumulator up front and reads it back when it emits `RunCompleted`, but
`InstrumentedRunnable.run` used to start a SECOND accumulator. Every LLM call accrued
into the second one, so the event — and the audit row the S4 subscriber writes from it —
carried zeros. `budget_usd` then summed a ledger of free runs and never fired.

These tests assert the numbers END TO END, through the shipped path, because that is
where the two accumulators diverged; asserting `last_metrics` alone missed it entirely.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import BaseModel

from lottie.core import BaseAgent
from lottie.governance.audit import SqliteAuditLogger
from lottie.governance.cost import BudgetExceeded, build_cost_gate
from lottie.llm import LLMResponse, Message
from lottie.llm.base import LLMProvider, TokenUsage


class _In(BaseModel):
    task: str


class _Out(BaseModel):
    answer: str


class _CostProvider(LLMProvider):
    """Reports a fixed cost and token count per call."""

    def __init__(self, cost: float) -> None:
        self._cost = cost
        self.calls = 0

    @property
    def model(self) -> str:
        return "cost/sim"

    def complete(
        self, messages: list[Message], model_params: Mapping[str, object] | None = None
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            content="ok",
            usage=TokenUsage(input_tokens=10, output_tokens=20),
            model="cost/sim",
            cost_usd=self._cost,
        )


class _Agent(BaseAgent[_In, _Out]):
    name = "CostProbe"

    def _execute(self, data: _In) -> _Out:
        return _Out(answer=self.complete([Message(role="user", content=data.task)]).content)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("LOTTIE_DISABLE_AUDIT", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _agent(cost: float, *, budget: float | None = None, root: Path | None = None) -> _Agent:
    agent = _Agent(llm=_CostProvider(cost), name="CostProbe", enable_benchmarks=False)
    if budget is not None and root is not None:
        agent.set_cost_gate(
            build_cost_gate(root, agent="CostProbe", budget_usd=budget)
        )
    return agent


class TestTheLedgerRecordsWhatTheRunSpent:
    def test_the_audit_row_carries_the_run_cost(self, root: Path) -> None:
        _agent(0.04).run(_In(task="hi"))
        rows = SqliteAuditLogger(root).query(limit=5)
        assert rows and rows[0].cost_usd == pytest.approx(0.04)

    def test_the_audit_row_carries_the_run_tokens(self, root: Path) -> None:
        _agent(0.04).run(_In(task="hi"))
        row = SqliteAuditLogger(root).query(limit=5)[0]
        assert (row.input_tokens, row.output_tokens) == (10, 20)

    def test_total_cost_accumulates_across_runs(self, root: Path) -> None:
        agent = _agent(0.04)
        for _ in range(3):
            agent.run(_In(task="hi"))
        assert SqliteAuditLogger(root).total_cost("CostProbe") == pytest.approx(0.12)

    def test_the_ledger_agrees_with_the_agent_metrics(self, root: Path) -> None:
        # The two accumulators used to disagree, and only one of them was checked.
        agent = _agent(0.04)
        agent.run(_In(task="hi"))
        assert agent.last_metrics is not None
        row = SqliteAuditLogger(root).query(limit=5)[0]
        assert row.cost_usd == pytest.approx(agent.last_metrics.cost_usd)

    def test_a_free_provider_still_records_zero(self, root: Path) -> None:
        _agent(0.0).run(_In(task="hi"))
        assert SqliteAuditLogger(root).query(limit=5)[0].cost_usd == pytest.approx(0.0)


class TestTheCumulativeBreakerFires:
    def test_the_run_that_crosses_the_budget_completes(self, root: Path) -> None:
        # One-run overshoot under sequential execution is the specified behaviour.
        agent = _agent(0.04, budget=0.05, root=root)
        agent.run(_In(task="hi"))
        agent.run(_In(task="hi"))
        assert SqliteAuditLogger(root).total_cost("CostProbe") == pytest.approx(0.08)

    def test_the_next_run_is_blocked(self, root: Path) -> None:
        agent = _agent(0.04, budget=0.05, root=root)
        agent.run(_In(task="hi"))
        agent.run(_In(task="hi"))
        with pytest.raises(BudgetExceeded):
            agent.run(_In(task="hi"))

    def test_a_blocked_run_never_reaches_the_provider(self, root: Path) -> None:
        agent = _agent(0.04, budget=0.05, root=root)
        agent.run(_In(task="hi"))
        agent.run(_In(task="hi"))
        provider = agent.llm
        assert isinstance(provider, _CostProvider)
        before = provider.calls
        with pytest.raises(BudgetExceeded):
            agent.run(_In(task="hi"))
        assert provider.calls == before

    def test_an_unbudgeted_agent_is_never_blocked(self, root: Path) -> None:
        agent = _agent(0.04)
        for _ in range(5):
            agent.run(_In(task="hi"))
        assert SqliteAuditLogger(root).total_cost("CostProbe") == pytest.approx(0.20)


class TestTheAccumulatorDoesNotLeak:
    def test_it_is_cleared_after_a_successful_run(self, root: Path) -> None:
        agent = _agent(0.04)
        agent.run(_In(task="hi"))
        assert agent._active_ctx is None

    def test_it_is_cleared_after_a_blocked_run(self, root: Path) -> None:
        # It is published BEFORE the chain runs, so a gate aborting ahead of `_execute`
        # would otherwise leave it dangling into the next call.
        agent = _agent(0.04, budget=0.01, root=root)
        agent.run(_In(task="hi"))  # post-hoc accrual: the first run is what crosses
        with pytest.raises(BudgetExceeded):
            agent.run(_In(task="hi"))
        assert agent._active_ctx is None

    def test_two_runs_do_not_share_an_accumulator(self, root: Path) -> None:
        agent = _agent(0.04)
        agent.run(_In(task="hi"))
        agent.run(_In(task="hi"))
        rows = SqliteAuditLogger(root).query(limit=5)
        assert [r.cost_usd for r in rows] == [pytest.approx(0.04), pytest.approx(0.04)]
