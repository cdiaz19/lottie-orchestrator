from __future__ import annotations

from pathlib import Path

import pytest

from lottie.governance.audit import SqliteAuditLogger
from lottie.governance.cost import (
    BudgetExceeded,
    CostGate,
    NullCostGate,
    build_cost_gate,
)
from lottie.governance.schema import AuditRecord


def _seed(root: Path, agent: str, cost: float) -> SqliteAuditLogger:
    logger = SqliteAuditLogger(root)
    logger.log(
        AuditRecord(
            ts="2026-06-14T10:00:00+00:00", agent=agent, provider="mock/x", status="ok",
            root=True, input_sha256="a" * 64, output_sha256="b" * 64, input_tokens=0,
            output_tokens=0, cost_usd=cost, latency_ms=1.0, error=None,
        )
    )
    return logger


def test_under_budget_passes(tmp_path: Path) -> None:
    ledger = _seed(tmp_path, "digest", 0.01)
    CostGate("digest", 0.10, ledger).check()  # no raise


def test_at_or_over_budget_blocks(tmp_path: Path) -> None:
    ledger = _seed(tmp_path, "digest", 0.10)
    with pytest.raises(BudgetExceeded):
        CostGate("digest", 0.10, ledger).check()  # spent == budget => block


def test_no_ledger_fails_closed() -> None:
    with pytest.raises(BudgetExceeded):
        CostGate("digest", 0.10, None).check()


def test_unreadable_ledger_fails_closed(tmp_path: Path) -> None:
    class _Boom(SqliteAuditLogger):
        def total_cost(self, agent: str) -> float:
            raise RuntimeError("db gone")

    with pytest.raises(BudgetExceeded):
        CostGate("digest", 0.10, _Boom(tmp_path)).check()


def test_null_gate_never_raises() -> None:
    NullCostGate().check()


def test_build_gate_no_budget_is_null(tmp_path: Path) -> None:
    assert isinstance(build_cost_gate(tmp_path, agent="x", budget_usd=None), NullCostGate)


def test_build_gate_with_budget_audit_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LOTTIE_DISABLE_AUDIT", raising=False)
    gate = build_cost_gate(tmp_path, agent="x", budget_usd=1.0)
    assert isinstance(gate, CostGate) and not isinstance(gate, NullCostGate)


def test_build_gate_audit_disabled_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOTTIE_DISABLE_AUDIT", "1")
    gate = build_cost_gate(tmp_path, agent="x", budget_usd=1.0)
    with pytest.raises(BudgetExceeded):  # ledger is None => fail closed
        gate.check()
