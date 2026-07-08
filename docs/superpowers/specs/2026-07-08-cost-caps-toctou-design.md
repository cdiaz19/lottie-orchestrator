# S3 Design — Per-run token cap + TOCTOU-safe atomic cost reservation

- **Date:** 2026-07-08
- **Epic:** v1.0.0 hardening — slice S3. Lab round R17.
- **Closes:** per-run token caps + TOCTOU atomic reservation (context.md deferrals).

## 1. Goal

Two cost-governance gaps:
1. **No per-run token cap** — a single runaway run can burn unbounded tokens under a
   cumulative budget.
2. **TOCTOU** — `CostGate.check()` reads prior committed spend and admits; N concurrent runs
   all see `spent < budget` and all proceed, overshooting by up to (in-flight × run cost).

## 2. Design

### 2.1 Per-run token cap (`AgentConfig.max_run_tokens: int | None`)
Enforced mid-run: `BaseAgent.complete()` / `stream_complete()` accrue usage into the active
`RunContext`. After each accrual, if `input_tokens + output_tokens > max_run_tokens`, raise
`TokenCapExceeded` — aborting the run before the next LLM call. Default `None` = off.
Attached by `instantiate_agent` (`agent.set_run_limits(max_run_tokens=...)`).

### 2.2 Atomic cost reservation (`AgentConfig.max_run_usd: float | None`)
`max_run_usd` is the per-run cost ceiling **and** the amount reserved up front. Reservations
live in a **`reservations` table in the same `.lottie/audit.db`**, so committed spend and
outstanding reservations are summed under **one `BEGIN IMMEDIATE` transaction** — true
atomicity across connections/processes (sqlite file-level RESERVED lock). The audit *table*
stays append-only; `reservations` is a separate ephemeral table (INSERT + DELETE).

`SqliteAuditLogger`:
- `reserve(agent, amount, budget) -> int`: `BEGIN IMMEDIATE`; `committed = SUM(audit.cost_usd
  WHERE agent)`; `reserved = SUM(reservations.amount WHERE agent)`; if
  `committed + reserved + amount > budget` → `ROLLBACK`, raise `BudgetExceeded`; else INSERT a
  reservation row, `COMMIT`, return its id.
- `settle(reservation_id)`: `DELETE FROM reservations WHERE id = ?` (the real cost has by then
  landed in `audit` via the run's audit record). Best-effort delete (a leaked reservation must
  not permanently wedge a budget — see §4).

### 2.3 CostGate lifecycle: reserve → settle
`CostGate` gains `reserve() -> int | None` and `settle(handle)`:
- `ledger is None` (audit disabled) + a budget set → fail-closed (as today).
- `max_run_usd is None` → **legacy** cumulative check (`committed >= budget` → raise); returns
  `None` (nothing to settle). Overshoot bound documented (unchanged behaviour, back-compat).
- `max_run_usd` set → **atomic**: `ledger.reserve(agent, max_run_usd, budget)` → handle.
`NullCostGate.reserve()` returns `None`; `settle(None)` is a no-op.

### 2.4 BaseAgent.run wiring
`_pre_run_gates` currently: `policy.check()` then `cost.check()`. Change to: `policy.check()`
then `handle = cost.reserve()`. In `run()`, **settle the reservation in `finally`** (always
released — success, output-withhold, or crash) so a reservation never leaks. `run_stream`
mirrors it. The token cap needs no wiring beyond §2.1.

## 3. Tests (grow from 903)
- `reserve`/`settle`: admits under budget; blocks when `committed + reserved + amount > budget`;
  a second concurrent reservation (simulated: reserve twice without settling) is refused →
  **TOCTOU closed**; settle frees headroom; reservation on a disabled ledger fails closed.
- Token cap: a run exceeding `max_run_tokens` raises `TokenCapExceeded` mid-run (before the next
  `complete`); under the cap runs clean; `None` = unlimited.
- Legacy path unchanged (budget_usd set, max_run_usd None → cumulative check; existing cost
  tests stay green).
- BaseAgent: reservation settled on success AND on failure (no leak); over-reserved run blocked
  and audited `budget_exceeded`.
- `instantiate_agent` attaches both new limits from config.

## 4. Risks / decisions
- **Leaked reservation:** always `settle` in `finally`. A process killed mid-run could orphan a
  row; mitigate with a coarse TTL sweep on read (`reserve` also deletes reservations older than
  a generous cutoff before summing). Keeps a crash from permanently shrinking a budget.
- **Reservation amount = `max_run_usd`** (pessimistic hold). Actual run cost may be lower; the
  hold is released at settle and the real cost recorded in audit. Overshoot now bounded to **0**
  for reserved runs (admission is atomic), vs (in-flight × cost) before.
- Same-connection multi-thread is NOT the target; per-agent `SqliteAuditLogger` = one connection
  each, `BEGIN IMMEDIATE` serializes across connections/processes at the file.
- `budget_usd` set without `max_run_usd` keeps the legacy (documented-overshoot) behaviour — no
  forced migration.

## 5. Files
- Edit: `governance/audit.py` (reservations table + `reserve`/`settle` + TTL sweep),
  `governance/cost.py` (`TokenCapExceeded`, `reserve`/`settle`, `build_cost_gate` max_run_usd),
  `core/base_agent.py` (reserve/settle in run + run_stream; token cap in complete/stream_complete;
  `set_run_limits`), `project/config.py` (`max_run_tokens`, `max_run_usd`),
  `project/discovery.py` (attach). Tests alongside.
