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


class TokenCapExceeded(Exception):
    """A single run exceeded its per-run token cap (max_run_tokens). Raised mid-run."""


class CostGate:
    """Admits a run against an agent's budget, holding a reservation for the run's duration.

    Two modes:
      * ``max_run_usd`` set -> **atomic reservation** (TOCTOU-safe): reserve the per-run
        ceiling under a single transaction that counts committed spend AND outstanding
        reservations, so concurrent runs cannot all slip past the same headroom.
      * ``max_run_usd`` None -> **legacy** cumulative check (block on prior committed spend;
        overshoot bounded by in-flight count) — preserved for back-compat.
    Fail-closed when a budget is set but the ledger is unreadable.
    """

    def __init__(
        self,
        agent: str,
        budget_usd: float,
        ledger: SqliteAuditLogger | None,
        *,
        max_run_usd: float | None = None,
    ) -> None:
        self._agent = agent
        self._budget = budget_usd
        self._ledger = ledger  # None => no readable ledger => fail closed
        self._max_run_usd = max_run_usd

    def check(self) -> None:
        """Legacy cumulative check: raise BudgetExceeded if prior spend has reached budget."""
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

    def reserve(self) -> int | None:
        """Admit the run. Returns a reservation handle to settle later, or None when there is
        nothing to settle (legacy path). Raises BudgetExceeded (fail-closed) on refusal."""
        if self._ledger is None:
            raise BudgetExceeded(
                f"agent {self._agent!r} has budget ${self._budget:.4f} but the audit "
                f"ledger is disabled - cannot verify spend (fail-closed)"
            )
        if self._max_run_usd is None:
            self.check()  # legacy cumulative check; no reservation handle
            return None
        try:
            handle = self._ledger.reserve(self._agent, self._max_run_usd, self._budget)
        except Exception as exc:  # ledger unreadable/locked => fail closed
            raise BudgetExceeded(
                f"agent {self._agent!r}: could not reserve budget (fail-closed): {exc}"
            ) from exc
        if handle is None:
            raise BudgetExceeded(
                f"agent {self._agent!r}: reserving ${self._max_run_usd:.4f} would exceed "
                f"budget ${self._budget:.4f} (committed + reserved)"
            )
        return handle

    def settle(self, handle: int | None) -> None:
        """Release a reservation handle from reserve() (no-op for the legacy/None path)."""
        if handle is not None and self._ledger is not None:
            self._ledger.settle(handle)


class NullCostGate(CostGate):
    """No-op gate — the BaseAgent default and the 'no budget declared' result."""

    def __init__(self) -> None:
        super().__init__("", 0.0, None)

    def check(self) -> None:
        return

    def reserve(self) -> int | None:
        return None

    def settle(self, handle: int | None) -> None:
        return


def build_cost_gate(
    root: Path, *, agent: str, budget_usd: float | None, max_run_usd: float | None = None
) -> CostGate:
    """NullCostGate when no budget is declared, else a CostGate bound to the audit ledger.

    The ledger is a real SqliteAuditLogger only when audit is enabled; when audit is
    disabled (LOTTIE_DISABLE_AUDIT / NullAuditLogger) the ledger is None and the gate
    fails closed. ``max_run_usd`` enables the atomic (TOCTOU-safe) reservation path.
    """
    if budget_usd is None:
        return NullCostGate()
    audit = build_audit_logger(root)
    ledger = audit if isinstance(audit, SqliteAuditLogger) else None
    return CostGate(agent, budget_usd, ledger, max_run_usd=max_run_usd)
