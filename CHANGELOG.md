# Changelog

All notable changes to Lottie Orchestrator. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions are
[semver](https://semver.org/).

## [1.0.0] — 2026-07-08

**"Complete, secured, documented."** V1 hardens, secures, and documents everything that
already exists — no new capabilities. Self-learning and agent-to-agent (A2A) are V2.

Delivered as a sliced epic (S1–S6), each PR validated downstream by a `lottie-lab` round
(R15–R20) and a full regression (R21).

### Added — security & governance
- **Rule 11 — per-skill-call capability enforcement** (`governance/capability.py`). An agent may
  only call skills in its `config.yaml` `capabilities` list; an undeclared call is blocked
  fail-closed at `BaseSkill.run` via an `_execute`-scoped gate. Whitelist-when-nonempty
  (empty = no enforcement). Framework security skills stay exempt.
- **Security gate on the BaseAgent/CLI path** (rules 8 & 9). `lottie run` and direct
  `BaseAgent.run` now pass input through sanitize + injection-scan and output through
  validate + secret-scan — the same gate serve uses — fail-closed, without double-gating serve.
- **Per-run token cap + TOCTOU-safe atomic cost reservation.** `max_run_tokens` bounds a single
  run's tokens; `max_run_usd` reserves the per-run ceiling under one `BEGIN IMMEDIATE` SQLite
  transaction that counts committed spend + outstanding reservations, closing the concurrent
  check-then-act race. Fail-closed on a disabled ledger.
- **HTTP hardening** for `lottie serve --port` (all opt-in via env): API-key auth
  (`LOTTIE_API_KEYS`, Bearer + `X-API-Key`, constant-time, open-when-unset), per-identity
  token-bucket rate limiting (`LOTTIE_RATE_LIMIT_PER_MIN`), and `limit`/`offset` pagination on
  `/v1/agents` + `/v1/models` (absent limit returns all — no silent truncation).

### Added — HITL & agentic hygiene
- **HITL edited_input-on-approve.** Resuming a paused mesh with `edited_input` now applies the
  human-edited `MeshState` fields (`task`/`final`) to the checkpoint before the worker runs,
  with fail-closed validation (bad edit → 400).
- **Agentic-loop rails.** `max_turns` caps LLM completions per run (`TurnLimitExceeded`); an
  optional `BaseAgent._verify(data, output)` hook (default no-op) lets an agent assert
  post-conditions and fail-closed before an output leaves it.

### Added — tooling
- `lottie doctor` warns when the HTTP transport would run without auth/rate-limit
  (`LOTTIE_API_KEYS` / `LOTTIE_RATE_LIMIT_PER_MIN` unset).

### New `config.yaml` fields (all optional; defaults preserve prior behaviour)
| Field | Effect |
|---|---|
| `capabilities: [..]` | non-empty → per-skill-call whitelist (rule 11) |
| `budget_usd` | cumulative per-agent spend cap (existing) |
| `max_run_usd` | per-run cost ceiling + atomic reservation amount |
| `max_run_tokens` | per-run token cap |
| `max_turns` | per-run LLM-completion cap |

### New environment variables
| Var | Effect (unset = off) |
|---|---|
| `LOTTIE_API_KEYS` | comma-separated valid API keys for the HTTP transport |
| `LOTTIE_RATE_LIMIT_PER_MIN` | per-identity request cap on the HTTP transport |

### Upgrade notes (0.x → 1.0.0)
- **No breaking changes.** Every new control is opt-in; an unchanged project behaves exactly as
  before. To adopt them: declare `capabilities` to enforce rule 11; set `max_run_usd` /
  `max_run_tokens` / `max_turns` per agent; set `LOTTIE_API_KEYS` (and optionally
  `LOTTIE_RATE_LIMIT_PER_MIN`) before exposing `lottie serve --port` publicly.
- `lottie run` now applies the security gate — an injection/oversized input is refused (exit 2).
  This is a behaviour change only for inputs that were already policy-violating.
- No public API removals; no internal deprecations in this release.

### Tests
- Grew from 828 (pre-v1) to **944** with the local gate (`ruff` + `mypy --strict` + `pytest`
  under `--all-extras`) green on every slice.

## [0.4.0] and earlier
See `.private-journey/JOURNEY.md` (dev log) — Phase 0 (core), Phase 1 (knowledge), Phase 2/3
(agent mesh + LangGraph hardening), Phase 4 (MCP, OpenAI-compat, REST, durable resume, real
token streaming).
