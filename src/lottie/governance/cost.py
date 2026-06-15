"""Per-agent cumulative cost budget circuit-breaker.

Reads accrued spend from the audit ledger (SqliteAuditLogger.total_cost) and blocks
fail-closed when a configured budget is reached or unverifiable. Imports only stdlib +
governance.audit, so governance stays free of core/project deps (acyclic).
"""

from __future__ import annotations

from pathlib import Path

from lottie.governance.audit import SqliteAuditLogger, build_audit_logger


class BudgetExceeded(Exception):
    """A per-agent cumulative cost budget has been reached. Raised pre-run, fail-closed."""


class CostGate:
    """Blocks a run when an agent's prior recorded spend has reached its budget."""

    def __init__(self, agent: str, budget_usd: float, ledger: SqliteAuditLogger | None) -> None:
        self._agent = agent
        self._budget = budget_usd
        self._ledger = ledger  # None => no readable ledger => fail closed

    def check(self) -> None:
        """Raise BudgetExceeded if the budget is reached or unverifiable; else return."""
        if self._ledger is None:
            raise BudgetExceeded(
                f"agent {self._agent!r} has budget ${self._budget:.4f} but the audit "
                f"ledger is disabled - cannot verify spend (fail-closed)"
            )
        try:
            spent = self._ledger.total_cost(self._agent)
        except Exception as exc:  # ledger unreadable => fail closed, never read spend as 0
            raise BudgetExceeded(
                f"agent {self._agent!r}: audit ledger unreadable, cannot verify spend "
                f"(fail-closed): {exc}"
            ) from exc
        if spent >= self._budget:
            raise BudgetExceeded(
                f"agent {self._agent!r} reached its budget: spent ${spent:.4f} "
                f">= ${self._budget:.4f}"
            )


class NullCostGate(CostGate):
    """No-op gate — the BaseAgent default and the 'no budget declared' result."""

    def __init__(self) -> None:
        super().__init__("", 0.0, None)

    def check(self) -> None:
        return


def build_cost_gate(root: Path, *, agent: str, budget_usd: float | None) -> CostGate:
    """NullCostGate when no budget is declared, else a CostGate bound to the audit ledger.

    The ledger is a real SqliteAuditLogger only when audit is enabled; when audit is
    disabled (LOTTIE_DISABLE_AUDIT / NullAuditLogger) the ledger is None and
    CostGate.check() fails closed.
    """
    if budget_usd is None:
        return NullCostGate()
    audit = build_audit_logger(root)
    ledger = audit if isinstance(audit, SqliteAuditLogger) else None
    return CostGate(agent, budget_usd, ledger)
