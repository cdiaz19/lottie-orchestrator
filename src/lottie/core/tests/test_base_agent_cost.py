from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from lottie.core.base_agent import BaseAgent
from lottie.governance.audit import SqliteAuditLogger
from lottie.governance.cost import BudgetExceeded, CostGate, NullCostGate
from lottie.governance.policy import PolicyDenied, PolicyGate
from lottie.governance.schema import AuditRecord
from lottie.llm import MockLLMProvider


class _In(BaseModel):
    q: str


class _Out(BaseModel):
    a: str


class _Spy(BaseAgent[_In, _Out]):
    def __init__(self, llm: object, audit: object, name: str) -> None:
        super().__init__(llm, name=name, audit=audit)  # type: ignore[arg-type]
        self.ran = False

    def _execute(self, data: _In) -> _Out:
        self.ran = True
        return _Out(a=f"ok:{data.q}")


def _seed(logger: SqliteAuditLogger, agent: str, cost: float) -> None:
    logger.log(
        AuditRecord(
            ts="2026-06-14T10:00:00+00:00", agent=agent, provider="mock/x", status="ok",
            root=True, input_sha256="a" * 64, output_sha256="b" * 64, input_tokens=0,
            output_tokens=0, cost_usd=cost, latency_ms=1.0, error=None,
        )
    )


def test_over_budget_blocks_before_execute_and_audits(tmp_path: Path) -> None:
    logger = SqliteAuditLogger(tmp_path)
    agent = _Spy(MockLLMProvider(["x"]), logger, "digest")
    _seed(logger, "digest", 0.10)  # prior spend at budget
    agent.set_cost_gate(CostGate("digest", 0.10, logger))
    with pytest.raises(BudgetExceeded):
        agent.run(_In(q="hi"))
    assert agent.ran is False
    statuses = [r.status for r in SqliteAuditLogger(tmp_path).query(limit=20)]
    assert "budget_exceeded" in statuses


def test_under_budget_runs_normally(tmp_path: Path) -> None:
    logger = SqliteAuditLogger(tmp_path)
    agent = _Spy(MockLLMProvider(["x"]), logger, "digest")
    agent.set_cost_gate(CostGate("digest", 1.00, logger))  # no prior spend
    out = agent.run(_In(q="hi"))
    assert out.a == "ok:hi" and agent.ran is True


def test_default_cost_gate_is_null(tmp_path: Path) -> None:
    agent = _Spy(MockLLMProvider(["x"]), SqliteAuditLogger(tmp_path), "digest")
    assert isinstance(agent._cost, NullCostGate)


def test_policy_checked_before_budget(tmp_path: Path) -> None:
    # Agent both policy-denied AND over budget -> PolicyDenied wins (policy first).
    logger = SqliteAuditLogger(tmp_path)
    agent = _Spy(MockLLMProvider(["x"]), logger, "digest")
    _seed(logger, "digest", 0.10)
    agent.set_policy(PolicyGate(["shell"], allow=set(), deny={"shell"}, escalate=set()))
    agent.set_cost_gate(CostGate("digest", 0.10, logger))
    with pytest.raises(PolicyDenied):
        agent.run(_In(q="hi"))
