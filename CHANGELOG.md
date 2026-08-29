# Changelog

All notable changes to Lottie Orchestrator. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions are
[semver](https://semver.org/).

## [3.2.0] — 2026-08-11

**"The config stops lying."** E5 makes `providers.fallback` mean something, and closes
the two scope limits `v3.1.0` shipped with.

### Fixed

- **`providers.fallback` was declared but never read.** `lottie.yaml` has carried it
  since Phase 0; nothing consumed it. A user who set one got nothing — worse than an
  absent feature, because the config claimed a resilience property the runtime lacked and
  the failure mode was discovering that during an outage. `resolve_provider` now honours
  it at every call site.

### Added — provider routing

- **`RoutedProvider`** advances through a provider chain on **transient** failures only:
  rate limit, timeout, 5xx, connection error.
- **A content-policy refusal is never retried elsewhere.** Shopping a refused request to a
  second model would launder a provider's safety decision through a framework that
  advertises fail-closed gates. That is the one retry this codebase must never make.
- **Bad requests and auth errors fail fast** — they fail identically on the fallback, so
  retrying only doubles the spend.
- **`is_transient` defaults to False** for unrecognised exceptions, so a new error type
  from a provider SDK cannot silently widen the fallback surface. Classification is by
  exception *name*, so the module never imports litellm (rule 1).
- **Streaming falls back only before the first delta.** Once bytes have shipped, switching
  providers would splice two models' answers into one response — silently corrupt, worse
  than a clean failure.
- A fallback leaves **two traces**: a warning when it happens, and the audit record
  afterwards, since `RoutedProvider.model` reports whoever actually served.

### Added — context (closing v3.1.0's scope limits)

- **Knowledge is a droppable source.** `complete()` gained an optional `context:`
  parameter so an agent can declare retrieved material *separately from its task*.
  `ResearchAgent` previously concatenated chunks into the user message, which made
  knowledge inseparable from the query — the compiler had nothing it was allowed to give
  up, and an over-budget prompt could only be compacted by position. Over budget now, the
  query survives and the knowledge is dropped or summarised.
- **E4 absorbed compaction.** `memory/compaction.py` moved to `context/` — it imports only
  `llm.Message` and had no memory dependencies, having lived under `memory/` since V2 S5a
  framed it as part of the harness. The compiler is now the single shrink authority
  instead of two mechanisms running back to back.

### Changed

- Pinning is decided at two levels, deliberately: which **sources** survive the budget,
  then which **messages** within a surviving source. Source-pinning cannot discriminate
  once every remaining source is pinned; there, role is the right signal.

### Known deviation from the E5 design

The design doc said a fallback would emit `fallback_triggered` on the runtime event bus.
**It does not.** The provider is constructed before the agent that owns the bus, so wiring
it there needs the module orchestrator to own provider construction — a larger change than
this epic. The warning plus the audit record give real observability without contorting
the wiring; the bus event can come with E7 if it earns its place.

### Backward compatibility

`build_provider(model)` and `complete(messages)` keep their signatures. A project with no
`fallback` configured is not wrapped at all and behaves identically to 3.1.0.

## [3.1.0] — 2026-08-10

**"Assembly gets an authority."** E4 gives message assembly an ordering authority, a
cross-source budget, and provenance — and finishes the module extraction V3 started.

Two slices (S1–S2), each validated downstream by a `lottie-lab` round (R34–R35).

### Added

- **`lottie.context.ContextCompiler`** — sources emit in declared order, the token ceiling
  is applied across all of them, and `CompileResult.contributions` records what each source
  cost. That last one closes the provenance gap: it answers *"which source filled the
  window?"*, which is exactly the question when a prompt gets expensive.
- **Pinning moved from role to source.** S5a had to pin on `role == "system"` because role
  was the only signal available — but a knowledge block and the recall block are *both*
  system messages, and only the recall block is load-bearing (S2a's anti-poisoning
  contract). Pinning is a source property now.
- **A drop policy that can choose.** Over budget, the compiler gives up the
  **lowest-order** droppable source — furthest from the task, so least contextually
  relevant — summarises rather than drops when a summariser exists, and stops as soon as
  the prompt fits. Compaction could only ever summarise by position.

### Changed

- **Reflection is a module.** `_maybe_reflect` left `BaseAgent` for
  `memory/middleware.py`, joining recall and trajectory.
- **Memory tier follows origin.** A trajectory is an EPISODIC event, a reflection lesson a
  SEMANTIC note — derived once in the gateway call so neither module knows the taxonomy.

### A prediction that was wrong, corrected

V3 S5 recorded that reflection could not become a module because it re-enters the agent's
own `complete()` with hand-primed budget state, and said **E4's Context Compiler would
unblock it**. That was wrong. What reflection actually needed is a narrow `BudgetedCaller`
Protocol — *"an LLM call that counts against this run's budget"* — and nothing to do with
message assembly. It could have been done in V3 S5.

### Scope limits, stated rather than implied

- **Every source shipped today is pinned**, so the drop policy cannot yet shrink a real
  prompt — compaction still does that, now called once at the end of assembly rather than
  mid-way. The policy is implemented and exercised (lab R34 cases 5–8 use a synthetic
  droppable source), but the *"drop stale knowledge before recent turns"* win needs
  knowledge wired in as a droppable source.
- **The import metric.** V3's headline was `core/base_agent.py`: **6 subsystem imports → 1**.
  Counting only the subsystem edges V3 set out to reverse — `governance`, `llm`, `memory` —
  it is at **3**; the `security` edge is gone. Counting every package, it is 5, two of which
  (`runtime`, `context`) are kernel-side and are the *intended* direction. The remaining
  three are the `self.memory`/`self.llm` DI fields, the delta types, and constructing the
  modules — the last of which is what a registry-driven orchestrator addresses.

### Backward compatibility

`complete(messages)` keeps its signature, so no agent changed. A 3.0.0 project behaves
identically.

## [3.0.0] — 2026-08-05

**"A runtime, not a base class."** V3 extracts an execution kernel so cross-cutting
concerns are **registered modules** rather than hand-sequenced steps inside
`BaseAgent.run`.

Delivered as a sliced epic (S1–S7), each PR validated downstream by a `lottie-lab` round
(R28–R33) before merge.

### Added — the kernel

- **`lottie.runtime`** — an execution kernel with two primitives, matching the two
  semantics already present in the code:
  - **Middleware** (Lifecycle Hooks) — ordered, abort-capable, can span `_execute` with a
    `try/finally`. For fail-closed gates.
  - **EventBus** (Event Runtime) — a fail-open observation stream. A subscriber that
    raises is warned and swallowed; it can neither fail a run nor starve the next
    observer.
- **`ScopedMiddleware`** — the streaming form. The plain middleware contract cannot
  express streaming: when `nxt` returns a generator, its `finally` fires at generator
  *creation*, settling the budget before a single delta exists. A scope composes through
  `ExitStack`, whose `with` spans consumption.
- **`ModuleRegistry`** with order-conflict detection at registration.
- **Events carry scalars and sha256 hashes only.** The bus is an observation surface the
  Plugin SDK will open to third parties, so a raw payload on it would be an exfiltration
  channel. The digest shape is **verified**, not trusted — an echoing hasher is refused.

### Changed — one execution path

- **`BaseAgent.run` is one line over the chain.** Twelve hand-sequenced cross-cutting
  steps became mounted modules.
- **`run_stream` shares the same middleware instances** (the scoped subset).
  `_pre_run_gates` is deleted; the duplicated execution path is gone.
- **Modules moved to their owning subsystems.** Security (rules 8/9) to
  `security/middleware.py`; policy, cost, capability (rule 11) to
  `governance/middleware.py`; recall and trajectory to `memory/middleware.py`; session to
  `session/middleware.py`. Each is constructed from its gate or client — none knows
  `BaseAgent`.
- **Auditing is an event subscriber.** Best-effort stopped being a `try/except` each
  observer had to remember and became a property of the bus.

### Added — operator surface

- **`lottie modules [<agent>]`** — the mounted chain, in execution order, with disabled
  modules shown explicitly. The chain was previously invisible.
- **A `modules:` config block** to switch a module off per agent. Built-in modules keep
  their existing top-level keys rather than growing a second way to configure the same
  thing.
- **`lottie doctor`** flags an unknown module name (a typo there does nothing, which is
  the dangerous kind of nothing) and warns loudly when a **fail-closed** module is
  disabled.

### Ordering, which turned out to be the hard part

Reproducing `run()`'s sequence exactly forced three findings, each of which would have
changed behaviour silently:

- **`CAPABILITY` is the innermost module.** The rule-11 gate is released *before*
  `_verify`, and `_verify` is user code that may call a skill.
- **`DEPTH` sits above `COST`.** `_write_block` reads `_depth() == 0` for the audit root
  flag; incrementing first would record a denied top-level run as a nested worker.
- **`run()`'s interleaving cannot be a pure onion** — the pre phase needs `COST < DEPTH`
  while the post phase would need the reverse. Two deviations are accepted and
  documented, both verified unobservable: `CostGate.settle` never reads the depth, and
  `_write_audit` received `is_root` as a captured parameter.

### Fixed

- **A cancelled stream was audited `status="ok"`.** `GeneratorExit` is a `BaseException`,
  so an `except Exception` missed it. Caught by a pre-existing test.
- **Audit records silently lost their `provider`.** Caught by lab R31.
- Removed dead code that the migration stranded: `_pre_run_gates`, `_write_audit`.

### Not delivered — stated plainly

The epic's headline metric was **`core/base_agent.py`: 6 subsystem imports → 1.** It is at
**5**. The middleware genuinely moved and are lab-proven to be owned by their subsystems,
but `base_agent` still imports them *to construct* them, plus `memory.compaction` and
`memory.reflection`.

Two things block the rest, both recorded rather than worked around:

- **Reflection did not become a module.** `_maybe_reflect` re-enters the agent's own
  `complete()` with hand-primed budget state; extracting it needs a Protocol that is
  `BaseAgent` in all but spelling.
- **Compaction is E4's by design** (spec §1.1) — it is a per-completion concern, not a
  run-lifecycle step.

Both close in **E4 (Context Compiler, v3.1)**, when message assembly and the run budget
become modules in their own right. The import reversal is on track, not complete.

### Backward compatibility

`run()` and `run_stream()` keep their signatures, exception types, and audit record
shapes. Every new capability is opt-in config. A 2.0.0 project with no config change
behaves exactly as before.

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
