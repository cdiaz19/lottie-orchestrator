# Governance — Cost Budget Circuit-Breaker (slice 3) — Design

> A per-agent cumulative spend cap. Before a run, the gate sums the agent's prior recorded cost
> from the audit ledger; if it has already reached `budget_usd`, the run is blocked **fail-closed**
> with a typed `BudgetExceeded`, recorded in the audit trail. A runaway agent stops spending.

- **Date:** 2026-06-14
- **Phase:** Governance, slice 3 (after audit #11 + policy #12). Cost budgets layered on the per-run
  `RunMetrics.cost_usd` already recorded by the audit trail.
- **Branch (this doc):** `docs/governance-cost-budget-spec` (docs-only).
- **Implementation dependency:** the audit root-flag parallel-worker fix
  (`fix/audit-root-parallel-workers`, commit `e99d42e`, **already merged to `main`**) should be on
  `main` before implementation — budgets read the audit ledger, and that fix keeps parallel-worker
  cost attribution correct. Satisfied.

---

## 1. Goal & scope

`LLMResponse.cost_usd` → `RunContext.add_usage` → `RunMetrics.cost_usd`, and the audit trail (slice 1)
already writes `cost_usd` per run into `.lottie/audit.db`. Nothing caps cumulative spend. This slice
adds a **cumulative budget circuit-breaker**: an agent that declares `budget_usd` is blocked from
starting a new run once its accumulated recorded spend has reached that budget.

**Locked decisions (do not relitigate):**
- **Mechanism:** cumulative budget circuit-breaker (per-run token caps are a separate later slice).
- **Scope:** **per-agent** — `budget_usd` in `config.yaml`.
- **Enforcement:** a pre-run gate at the `BaseAgent.run` chokepoint, the same seam the policy gate
  uses, attached via `instantiate_agent`.
- **Ledger:** reuse the audit trail (`.lottie/audit.db`) as the spend source.
- **Fail-closed:** a configured budget with no readable ledger blocks; it never reads spend as 0.
- **Ledger key:** agent **name** (the metrics/audit `name`, class-derived by default).
- **Per-project budget:** deferred (see §8).

## 2. Pre-run semantics — block on *prior* accrual

This run's cost is unknown before the LLM call, so the gate **cannot** pre-flight `accrued + this_run`.
The locked rule is therefore **block purely on prior cumulative spend**:

> Before a run, `prior = SUM(cost_usd) WHERE agent = <name>` from the audit db. If
> `prior >= budget_usd` → block (`BudgetExceeded`). Otherwise allow the run; its cost is recorded
> post-hoc by the audit hook and counts toward the *next* check.

Consequence — **one-run overshoot is allowed by design.** The run that *crosses* `budget_usd` completes
(it was under budget when it started); the *next* run is blocked. This is a post-hoc accrual circuit
breaker, not a pre-flight cost estimator. Documented, intentional, and YAGNI-correct for a first slice
(pre-flight estimation would need per-model token pricing + an input-token estimate — out of scope).

## 3. Spend source — `SqliteAuditLogger.total_cost`

Add one read method to the existing logger (`src/lottie/governance/audit.py`); no schema change
(`cost_usd` column already exists):

```python
class SqliteAuditLogger(AuditLogger):
    ...
    def total_cost(self, agent: str) -> float:
        """Sum of cost_usd across all recorded runs for `agent` (0.0 if none)."""
        conn = self._connect()
        row = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) FROM audit WHERE agent = ?", (agent,)
        ).fetchone()
        return float(row[0])
```

`total_cost` is a `SELECT` (read-only) — it preserves the append-only immutability guarantee
(`NullAuditLogger` has no such method; the cost gate treats a Null/absent ledger as "unverifiable" —
see §5). This is the only change to the audit module.

## 4. The gate — `src/lottie/governance/cost.py`

```python
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
                f"ledger is disabled — cannot verify spend (fail-closed)"
            )
        try:
            spent = self._ledger.total_cost(self._agent)
        except Exception as exc:  # ledger unreadable => fail closed, never read as 0
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


def build_cost_gate(
    root: Path, *, agent: str, budget_usd: float | None
) -> CostGate:
    """NullCostGate when no budget is declared, else a CostGate bound to the audit ledger.

    The ledger is a real SqliteAuditLogger only when audit is enabled; when audit is disabled
    (LOTTIE_DISABLE_AUDIT / NullAuditLogger) the ledger is None and CostGate.check() fails closed.
    """
    if budget_usd is None:
        return NullCostGate()
    audit = build_audit_logger(root)
    ledger = audit if isinstance(audit, SqliteAuditLogger) else None
    return CostGate(agent, budget_usd, ledger)
```

`governance.cost` imports stdlib + `governance.audit` only (no `core`/`project`) — acyclic, mirroring
`governance.policy`. `build_cost_gate` takes primitives (`agent`, `budget_usd`), not `AgentConfig`,
for the same layering reason the policy factory does.

## 5. Fail-closed behavior matrix

| `budget_usd` | audit ledger | prior spend | outcome |
|---|---|---|---|
| `None` (absent) | any | — | **allow** (`NullCostGate`, no check — backward-compatible) |
| set | enabled (`SqliteAuditLogger`) | `< budget` | **allow** |
| set | enabled | `>= budget` | **block** `BudgetExceeded` |
| set | **disabled** (`LOTTIE_DISABLE_AUDIT` / `NullAuditLogger`) | unverifiable | **block** `BudgetExceeded` (fail-closed) |
| set | enabled but query raises | unverifiable | **block** `BudgetExceeded` (fail-closed) |

The security property: a configured budget that cannot be verified **blocks**, exactly as the policy
engine fails closed on a malformed/unreadable policy. Spend is never silently assumed 0.

## 6. Enforcement — `BaseAgent.run` (`src/lottie/core/base_agent.py`)

`BaseAgent` gains `self._cost: CostGate = NullCostGate()` (default, set in `__init__`) and
`set_cost_gate(self, gate: CostGate) -> None`. The existing pre-check block runs the cost check
**after** the policy check:

```python
    def run(self, data: InputT) -> OutputT:
        try:
            self._policy.check()        # capability policy (slice 2) — checked FIRST
            self._cost.check()          # cumulative budget (this slice) — checked SECOND
        except PolicyViolation as exc:
            self._write_block(data, exc); raise
        except BudgetExceeded as exc:
            self._write_block(data, exc); raise
        # ... existing depth + super().run + _write_audit (unchanged) ...
```

**Order if both fire:** policy is checked first (it's the cheaper, no-I/O capability check; budget
needs an audit query). If an agent both violates policy *and* is over budget, it surfaces
`PolicyDenied`/`PolicyEscalation`, not `BudgetExceeded`. Stated explicitly.

**Blocked-run audit:** generalize the existing `_write_policy_block` into a `_write_block(data, exc)`
that maps the exception to a `status`: `PolicyDenied → "denied"`, `PolicyEscalation → "escalated"`,
`BudgetExceeded → "budget_exceeded"`. Same best-effort wrapper (record-building cannot break or mask
the block), `root=_depth() == 0`, zeroed metrics, `error=str(exc)`. `AuditRecord.status` vocabulary
widens to `ok | error | denied | escalated | budget_exceeded` (status is a free `str`; no schema
change). The check runs before `_execute`, so a blocked run never reaches the LLM — no spend incurred.

## 7. Config & wiring

- **`AgentConfig`** (`src/lottie/project/config.py`) gains `budget_usd: float | None = None`. Absent ⇒
  unlimited ⇒ backward-compatible (every existing agent is unchanged).
- **`instantiate_agent`** (`src/lottie/project/discovery.py`) — the seam both `lottie run` and
  `AgentService` use — attaches the gate after construction, alongside the existing policy attach:
  ```python
      agent.set_policy(build_policy_gate(root, policies=config.policies, capabilities=config.capabilities))
      agent.set_cost_gate(build_cost_gate(root, agent=agent.name, budget_usd=config.budget_usd))
      return agent
  ```
  The ledger key is `agent.name` (the constructed agent's metrics name) — the **same** name the audit
  rows are written under, so `total_cost(agent.name)` matches. Directly-constructed agents (tests,
  in-process) keep `NullCostGate` — no behavior change.

**Ledger-key limitation (accepted):** `agent.name` defaults to the class name unless a `name=` is
passed, so two instances of the same agent class share one budget. Documented; per-instance budgets
are out of scope.

**Ledger-root alignment (assumption to verify in implementation):** the cost gate reads
`root/.lottie/audit.db`, while an agent's own audit writes to `<benchmarks_root>/.lottie/audit.db`
(`benchmarks_root` defaults to `Path.cwd()`). For `lottie run` and `AgentService` both resolve to the
**project root**, so the two coincide. The implementation must confirm the gate's `root` is the project
root that the agent's audit also writes under; if a future caller diverges them, the budget would read
a different ledger. Note it; don't over-engineer a fix this slice.

## 8. Out of scope / deferred (YAGNI)

- **Per-run token/cost caps** (`max_tokens_per_run`, DoS clamp on `model_params`) — a separate later
  slice; different mechanism (pre-call clamp vs cumulative ledger).
- **Per-project budget** — DEFERRED. `MeshAgent` rolls worker cost into its own `RunMetrics` (via
  `_accumulate`) **and** each worker writes its own audit row, so `SUM(cost_usd)` over *all* rows
  double-counts mesh-worker spend. A correct project budget must first resolve that double-count
  (e.g. budget only `root=True` rows, or exclude worker rollups) — non-trivial, its own slice. Per-agent
  budgeting avoids this: each agent's rows are summed independently, no cross-agent rollup.
- **Daily / windowed budgets, reset, soft-warn thresholds** — all-time cumulative only this slice.
- **Pre-flight cost estimation** — the one-run overshoot (§2) is accepted instead of estimating this
  run's cost from input tokens × per-model pricing.
- **Forwarding `audit`/`cost` through `MeshAgent.__init__`** — `MeshAgent` still doesn't forward an
  injected logger/gate (noted in the audit slice); unchanged here.

## 9. Testing

- **`SqliteAuditLogger.total_cost`** (unit): sums `cost_usd` for an agent across rows; `0.0` for an
  unknown agent; ignores other agents' rows.
- **`CostGate.check`** (unit): `spent < budget` → no raise; `spent >= budget` → `BudgetExceeded`;
  `ledger is None` → `BudgetExceeded` (fail-closed); a ledger whose `total_cost` raises →
  `BudgetExceeded` (fail-closed); `NullCostGate` → never raises.
- **`build_cost_gate`** (unit): `budget_usd=None` → `NullCostGate`; budget set + audit enabled →
  `CostGate` with a real ledger; budget set + `LOTTIE_DISABLE_AUDIT` → `CostGate` with `ledger=None`
  (its `check` then fails closed).
- **`BaseAgent.run` enforcement** (integration, injected `SqliteAuditLogger(tmp)` + `set_cost_gate`):
  seed prior spend ≥ budget → `run` raises `BudgetExceeded`, `_execute` never ran (spy / `llm` index
  0), exactly one audit record `status="budget_exceeded"`; prior spend < budget → runs normally, row
  `status="ok"`; an agent both policy-denied and over budget → `PolicyDenied` (policy first); default
  `NullCostGate` → unaffected.
- **`instantiate_agent`** (integration): a scaffolded agent with `budget_usd` in config + seeded
  audit spend ≥ budget → `instantiate_agent(...).run(...)` raises `BudgetExceeded`; no `budget_usd` →
  runs.
- **Full gate:** `uv run pytest -q`, `uv run mypy --strict src`, `uv run ruff check` — all green;
  existing suite unaffected (no `budget_usd` anywhere ⇒ `NullCostGate` everywhere).
- **lottie-lab round (Round 8 case):** give the lab `DigestAgent` a small `budget_usd`; drive runs
  (real instantiate path + scripted/Mock provider with a non-zero cost, or pre-seeded audit rows)
  until accrued spend reaches the budget; assert the next run raises `BudgetExceeded`, is audited
  `status="budget_exceeded"`, and `lottie audit` shows the row; assert that with audit disabled a
  budgeted agent fails closed. (Mirrors the Round-7 governance harness.)

## 10. Definition of done

`budget_usd` declared on an agent caps its cumulative spend at the `BaseAgent.run` chokepoint:
`prior >= budget_usd` (read from `audit.db` via `total_cost`) → `BudgetExceeded` before `_execute`,
audited `status="budget_exceeded"`; fail-closed when the ledger is disabled/unreadable; policy checked
before budget; absent `budget_usd` ⇒ unlimited (backward-compatible); gate attached by
`instantiate_agent`; `NullCostGate` default unchanged. `governance.cost` acyclic. `pytest` / `mypy
--strict src` / `ruff` green. Per-project budget + per-run caps explicitly deferred (§8). Commit on the
feature branch; do not push until approved.
