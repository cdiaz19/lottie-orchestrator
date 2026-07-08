"""Atomic cost reservation (TOCTOU-safe admission) — reserve/settle on SqliteAuditLogger
and the CostGate reserve/settle lifecycle."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from lottie.governance.audit import SqliteAuditLogger
from lottie.governance.cost import BudgetExceeded, CostGate, NullCostGate, build_cost_gate
from lottie.governance.schema import AuditRecord


def _record(agent: str, cost: float) -> AuditRecord:
    return AuditRecord(
        ts=datetime.now(UTC).isoformat(), agent=agent, provider="p", status="ok", root=True,
        input_sha256="x", output_sha256="y", input_tokens=1, output_tokens=1,
        cost_usd=cost, latency_ms=1.0, error=None,
    )


class TestLedgerReserve:
    def test_admits_under_budget(self, tmp_path: Path) -> None:
        lg = SqliteAuditLogger(tmp_path)
        assert lg.reserve("a", amount=3.0, budget=10.0) is not None

    def test_blocks_when_over_budget(self, tmp_path: Path) -> None:
        lg = SqliteAuditLogger(tmp_path)
        assert lg.reserve("a", amount=11.0, budget=10.0) is None

    def test_second_unsettled_reservation_refused_toctou(self, tmp_path: Path) -> None:
        lg = SqliteAuditLogger(tmp_path)
        first = lg.reserve("a", amount=6.0, budget=10.0)
        assert first is not None
        # committed=0, reserved=6, +6 = 12 > 10 -> the concurrent run is refused
        assert lg.reserve("a", amount=6.0, budget=10.0) is None

    def test_settle_frees_headroom(self, tmp_path: Path) -> None:
        lg = SqliteAuditLogger(tmp_path)
        first = lg.reserve("a", amount=6.0, budget=10.0)
        assert first is not None
        lg.settle(first)
        assert lg.reserve("a", amount=6.0, budget=10.0) is not None

    def test_committed_spend_counts_against_reservation(self, tmp_path: Path) -> None:
        lg = SqliteAuditLogger(tmp_path)
        lg.log(_record("a", 7.0))  # committed 7
        assert lg.reserve("a", amount=4.0, budget=10.0) is None  # 7+4 > 10
        assert lg.reserve("a", amount=3.0, budget=10.0) is not None  # 7+3 == 10, admitted

    def test_reservations_are_per_agent(self, tmp_path: Path) -> None:
        lg = SqliteAuditLogger(tmp_path)
        assert lg.reserve("a", amount=9.0, budget=10.0) is not None
        assert lg.reserve("b", amount=9.0, budget=10.0) is not None  # different agent


class TestCostGateReserve:
    def test_atomic_reserve_returns_handle(self, tmp_path: Path) -> None:
        gate = CostGate("a", 10.0, SqliteAuditLogger(tmp_path), max_run_usd=4.0)
        handle = gate.reserve()
        assert handle is not None
        gate.settle(handle)

    def test_atomic_reserve_blocks_over_budget(self, tmp_path: Path) -> None:
        lg = SqliteAuditLogger(tmp_path)
        gate = CostGate("a", 10.0, lg, max_run_usd=6.0)
        assert gate.reserve() is not None
        with pytest.raises(BudgetExceeded):
            gate.reserve()  # second run's reservation would exceed budget

    def test_legacy_reserve_no_handle(self, tmp_path: Path) -> None:
        lg = SqliteAuditLogger(tmp_path)
        lg.log(_record("a", 5.0))
        gate = CostGate("a", 10.0, lg, max_run_usd=None)  # legacy cumulative
        assert gate.reserve() is None  # spent 5 < 10, admitted, nothing to settle

    def test_legacy_reserve_blocks_when_spent_reaches_budget(self, tmp_path: Path) -> None:
        lg = SqliteAuditLogger(tmp_path)
        lg.log(_record("a", 10.0))
        gate = CostGate("a", 10.0, lg, max_run_usd=None)
        with pytest.raises(BudgetExceeded):
            gate.reserve()

    def test_reserve_fails_closed_without_ledger(self) -> None:
        gate = CostGate("a", 10.0, None, max_run_usd=4.0)
        with pytest.raises(BudgetExceeded):
            gate.reserve()

    def test_null_gate_reserve_and_settle_noop(self) -> None:
        gate = NullCostGate()
        assert gate.reserve() is None
        gate.settle(None)

    def test_build_cost_gate_wires_max_run_usd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # audit is globally disabled in the test suite; enable it so build_cost_gate binds a
        # real ledger and the reservation path is exercised end-to-end.
        monkeypatch.delenv("LOTTIE_DISABLE_AUDIT", raising=False)
        gate = build_cost_gate(tmp_path, agent="a", budget_usd=10.0, max_run_usd=4.0)
        assert isinstance(gate, CostGate) and not isinstance(gate, NullCostGate)
        assert gate.reserve() is not None
