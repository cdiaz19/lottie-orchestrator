# S3 Plan — cost caps + TOCTOU reservation

Design: `2026-07-08-cost-caps-toctou-design.md`. TDD, mypy --strict + ruff clean per task. Baseline 903.

- **T1** `audit.py`: `reservations` table (id, agent, amount, ts); `reserve(agent, amount, budget) -> int` (BEGIN IMMEDIATE, sum committed+reserved, block-or-insert, TTL sweep of stale rows first); `settle(id)`. Tests: admit under budget; block when committed+reserved+amount>budget; second unsettled reserve refused (TOCTOU); settle frees headroom.
- **T2** `cost.py`: `TokenCapExceeded`; `CostGate.reserve()/settle()` (legacy check when max_run_usd None; atomic when set; fail-closed when ledger None); `NullCostGate` no-ops; `build_cost_gate(..., max_run_usd)`. Keep `check()` for back-compat/legacy. Tests.
- **T3** `config.py`: `max_run_tokens: int | None`, `max_run_usd: float | None`.
- **T4** `base_agent.py`: `set_run_limits(max_run_tokens=None)`; token cap in `complete`/`stream_complete` (after add_usage, raise `TokenCapExceeded` if in+out tokens > cap); reserve in `_pre_run_gates` (replace `cost.check()` with `handle = cost.reserve()`), settle in `run`/`run_stream` `finally`. Tests: settle on success+failure (no leak); over-reserve blocked+audited `budget_exceeded`; token cap mid-run.
- **T5** `discovery.py`: attach `max_run_usd` to the cost gate + `set_run_limits(max_run_tokens)`.
- **T6** Full gate + whole-diff review.

## Lab R17
Downstream: budget with max_run_usd → atomic admission (a second concurrent unsettled reservation refused); per-run token cap aborts a runaway run; legacy budget still works; fail-closed on disabled ledger.
