# Changelog

All notable changes to Lottie Orchestrator. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions are
[semver](https://semver.org/).

## [2.0.0] — 2026-08-03

**"Agents that learn, runs that outlive the process."** V2 closes the two gaps against the
mid-2026 state of the art — self-learning agents and long-running harness ergonomics —
while staying provider-agnostic, typed, governed, and fail-closed.

Delivered as a sliced epic (S0–S6), each PR validated downstream by a `lottie-lab` round
(R22–R27) before merge.

### Added — self-learning

- **Persistent memory store** (S0). `SqliteMemoryClient` over `.lottie/memory.db`, with
  provenance and lifecycle on every record, an incremental `update`/`MemoryPatch` op, and
  a `build_memory_client` factory wired through `instantiate_agent`.
- **Write gateway and poisoning defence** (S1, rule 13b). `MemoryAgent.apply` is the only
  path for learned content: every delta is injection- and secret-screened fail-closed,
  deduped, provenance-stamped, and audit-trailed hash-only. Soft-deprecate, never delete.
- **Recall as data** (S2a). Recalled notes reach the model inside a `render_as_data` block
  with the delimiter defanged — a run cannot write instructions that hijack a future run.
- **Reflexive write-back** (S2b). An opt-in post-run hook distils a run into lessons
  through the gateway, budget-counted and best-effort. `lottie reflect` does it manually.
- **Episodic trajectory persistence** (S3a). Each run can be appended to the episodic tier,
  giving `lottie reflect` and distillation a corpus. Spends no tokens.
- **Skill distillation** (S3b, rule 13c). `lottie distill run` turns successful trajectories
  into a **parameterized prompt template** — never generated Python. Executed by one generic
  `TemplateRunnerSkill`; nothing an LLM authored is ever imported or executed.
- **Human promotion** (S3c). `lottie distill review --approve` re-screens the draft, moves
  it to `skills/distilled/`, and records who approved it under which capability. An agent
  must declare both `distilled` and that capability.
- **Learning-delta benchmark** (S4). `lottie benchmark agent <name> --learning-delta` runs
  the suite with recall off, then on, and reports seven metric deltas plus a verdict. Both
  arms disable memory writes, so the measurement never mutates what it measures.

### Added — long-running harness

- **Context compaction** (S5a). Older turns are summarised into one `[compacted history]`
  message when a run nears its window. System messages are pinned, so the recall-as-data
  block always survives. Opt-in via `harness.compaction`.
- **Session artifacts** (S5b). `lottie run <agent> --session <id>` resumes earlier progress;
  agents read `session_progress` and call `save_progress`. Progress persists on every call,
  so a run that dies halfway keeps what it achieved. Run history is hash-only. Adds
  `lottie session list|show|delete`.

### Fixed

- **Injection-scanner role-spoofing bypasses.** Found by the R23 memory-poisoning red-team.
  `SYSTEM:` role prefixes, ChatML `<|im_start|>` control tokens, jailbreak-mode phrasing,
  and `disregard prior` all reached the memory write gateway unflagged. Three new rules plus
  a broadened one, with false-positive guards.
- **`lottie reflect` was a structural no-op** — it consolidates episodic→semantic, but
  nothing ever wrote episodic records. S3a gives it input.

### Changed

- `MemoryAgent.apply` takes a `tier` argument (default `SEMANTIC`, so existing callers are
  unchanged). Episodic writes are append-only and skip content dedup.
- The three-scanner content screen is shared as `security/content_gate.ContentGate`;
  `MemoryContentGate` keeps its public API as a thin subclass.

### Security notes

- **Learning is OFF by default and stays that way.** See "default-on decision" below.
- Trajectories store raw task/outcome text where the audit ledger stores only hashes. They
  are gated on write, size-bounded, and confined to the EPISODIC tier that recall never
  reads. A project handling sensitive input should leave `memory.trajectory` off.
- Distilled templates and session progress are both screened on write, because both
  round-trip into a future run.
- `lottie doctor` gains advisories for unbounded reflection spend and for trajectories that
  are written but never consulted.

### The default-on decision

The epic made turning write-back on by default conditional on the learning-delta benchmark
showing a **non-negative** result. **The decision is: learning stays opt-in.**

The benchmark machinery is complete and honest — it reports `improved`/`neutral`/`regressed`
from the accuracy delta and states how many notes were recalled, so a neutral verdict over
an empty store is distinguishable from learning genuinely not helping. What is missing is
evidence: a real-LLM eval run across a populated store. Until that exists there is no basis
for flipping the default, and shipping a default-on behaviour that spends tokens without
demonstrated benefit would be the wrong trade. Re-open this when a real-model delta report
exists.

### Backward compatibility

Every new capability is opt-in config; a project upgrading from 1.0.0 with no config change
behaves exactly as before. No public API was removed.

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
