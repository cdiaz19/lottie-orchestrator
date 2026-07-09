# Lottie V2 (Phase 5) — Self-Learning Agents & Long-Running Harness

> Epic design. Target: **v2.0.0**. Date: 2026-07-08.
> Theme: close the two gaps vs the mid-2026 state of the art — **self-learning agents**
> and **long-running harness ergonomics** — while staying provider-agnostic, typed,
> governed, and fail-closed.

---

## 1. Context & motivation

V1 (v1.0.0, 2026-07-08) hardened, secured, and documented everything that already
existed — no new capabilities. V2 adds capability along two axes:

1. **Self-learning** — agents distil lessons from their own execution into memory and
   into reusable skills, measurably improving over time.
2. **Long-running ergonomics** — a run can exceed one context window and one process:
   context compaction, session artifacts, and multi-session progress.

**Load-bearing precondition discovered during brainstorming:** the memory subsystem is
still stubs. `memory/base.py` ships `MemoryClient` (ABC), `NullMemoryClient` (fail-loud),
`MockMemoryClient` only; no persistent store exists (context.md notes real
SQLite/Chroma/YAML stores as *Deferred*). `MemoryClient` exposes `remember`/`recall`/
`forget` — **no incremental `update`**, which ACE-style playbook evolution requires.
`MemoryAgent._execute` today writes straight through `self.memory.remember` with no
SecurityGate, no audit, no dedup. So "MemoryAgent is the mandatory write gateway" is a
goal, not current reality. V2 therefore builds a real store and a hardened gateway
*before* any reflection behavior.

**Inspiration:** ACE (arXiv 2510.04618) — memory as an evolving playbook, updated by
**incremental structured deltas**, never wholesale rewrites, with dedup/curation so it
does not bloat.

---

## 2. Decisions (settled in brainstorming)

| # | Decision | Rationale |
|---|---|---|
| D1 | **Store: SQLite, structured/tag recall.** Add an `update`/patch op to `MemoryClient`. Defer ChromaDB/vector recall. | Smallest surface that makes write-back real + measurable; honors rule 16 (Chroma only >~200 files). |
| D2 | **Distilled skill = parameterized prompt template, zero codegen.** Executed by a generic template-runner skill. | No LLM-authored Python ever executes — sidesteps rule-13 codegen pipeline + arbitrary-exec risk. Still typed, versioned, provenance-tagged, HITL-gated. |
| D3 | **Reflection = opt-in post-run hook + `lottie reflect` CLI; Reflector emits ACE-style delta ops** applied through MemoryAgent. | Matches "self-learning after each run" while keeping it OFF by default, budgeted, and observable. |
| D4 | **Epic shape: granular, dependency-first (~7 slices), one PR each.** | Mirrors the V1 cadence that stayed green; isolates a poisoning bug to one small PR. |

---

## 3. Architecture

### 3.1 Memory data model (extends `memory/schema.py`)

- `MemoryRecord` gains provenance + lifecycle: `origin: reflection|distill|manual`,
  `source_agent: str`, `run_id: str | None`, `status: active|deprecated`,
  `created_at`/`updated_at`.
- New `MemoryPatch` — all-optional fields for incremental `update` (content, tags,
  status, metadata). Never a wholesale replace.
- New `MemoryDelta(op: ADD|UPDATE|DEPRECATE, target_id: str | None, content: str,
  tags: list[str])` — the unit the Reflector emits and MemoryAgent applies.

### 3.2 Store (`memory/store.py`)

`SqliteMemoryClient(MemoryClient)` → `.lottie/memory.db`, single `records` table keyed by
`memory_id`, filtered by `namespace`/`tier`/`tags`/`status`. Structured recall only
(deterministic recency/tag-overlap score). `build_memory_client(config)` factory mirrors
`build_provider`/`build_audit_logger`; `instantiate_agent` injects `self.memory`.
Disabled/unreadable ledger → `NullMemoryClient` (fail-closed).

### 3.3 Write gateway (`MemoryAgent.apply`)

MemoryAgent is the **only** persistence path for learned content. `apply(list[MemoryDelta])`:
1. Runs each delta's content through the SecurityGate **input** path (InputSanitizer +
   PromptInjectionScan + SecretDetection). Injection-like/secret content → rejected
   fail-closed, audited `status="memory_rejected"`.
2. Stamps provenance (run_id / source_agent / origin).
3. Applies incrementally — ADD (dedup by content-hash; near-dup folds to UPDATE),
   UPDATE (patch in place), DEPRECATE (soft `status`, never hard-delete).
4. Audit-trails every write (new `memory_write` action, hash only — never raw content).

### 3.4 Reflection (`Reflector` + BaseAgent hook)

BaseAgent accumulates a `RunTrajectory` (turns, tool/skill results, errors, gate verdicts,
`RunMetrics`, final output) in-memory during `_execute` — **only materialized when
reflection is enabled** (zero cost otherwise). A post-run hook in `BaseAgent.run` (mirrors
the audit hook) invokes the `Reflector` skill → `list[MemoryDelta]` → `MemoryAgent.apply`
when `memory.reflect.enabled`. `lottie reflect <agent>` does the same manually/batched.

### 3.5 Recall-as-data

Recalled memory is returned tagged **as data** with its provenance. Consuming agents
render it as context, never as system instructions. This is the anti-poisoning contract:
a run cannot write instructions that hijack a future run.

### 3.6 Distillation (template skills)

`lottie distill <agent>` selects successful trajectories → LLM authors a **template-skill
draft** in `skills/draft/<name>/` (`SKILL.md` + `template.yaml` {typed prompt with slots,
Input/Output schema names} + `provenance.yaml` {producing run ids, version}). A generic
`TemplateRunnerSkill(BaseSkill)` executes a `DistilledSkill` (fill typed slots → LLM →
validate output). Draft content is untrusted → SecretDetection + PromptInjectionScan before
write. `lottie distill review` promotes draft→registered (HITL, reuses the knowledge
draft→curated pattern); capability declared at promotion. Versioned (semver) + provenance;
re-distill bumps version. **No Python authored or executed.**

### 3.7 Long-running harness

- **Context compaction** in BaseAgent — near a token threshold, summarize older turns,
  keep recent N + load-bearing (system/task). Opt-in `harness.compaction`, budgeted, OFF.
- **Session artifacts** — `SessionStore` persists run state (turns/summary/progress) to
  `.lottie/sessions/<id>/`, generalizing the #17 durable-resume machinery from mesh to
  plain BaseAgent runs.
- **Initializer / incremental progress** — `_init_session`/`_load_progress`/`_save_progress`
  hooks + `lottie run <agent> --session <id>` resume.

### 3.8 Self-improvement eval loop

`lottie benchmark <agent> --learning-delta` runs a task suite twice — **baseline** (recall
+ distilled skills disabled, clean/read-only namespace) vs **learning** (enabled, populated
namespace) — and reports per-metric delta (quality, tokens, cost, latency) + aggregate to a
machine-readable report in `.lottie/benchmarks/`. This report **gates the default-on
decision** for write-back. Real-LLM = eval tier.

---

## 4. Slice plan (risk/dependency-first; one PR each)

| Slice | Delivers | Gated by | Lab |
|---|---|---|---|
| **S0 Store foundation** | `SqliteMemoryClient`, `update`/`MemoryPatch`, provenance+status on `MemoryRecord`, `build_memory_client`, inject via `instantiate_agent` | — | R22 |
| **S1 Gateway + poisoning spine** | `MemoryAgent.apply(deltas)`, SecurityGate-on-write, `memory_write` audit, provenance, dedup + ADD/UPDATE/DEPRECATE, recall-as-data | S0 | R23 (red-team) |
| **S2 Reflexive write-back** | `RunTrajectory` capture, `Reflector`, opt-in post-run hook + `lottie reflect`, budget-counted, OTel span, recall injection | S1 | R24 |
| **S3 Distillation (template)** | `lottie distill` → draft template-skill, `TemplateRunnerSkill`, `lottie distill review` HITL promote, version+provenance | S1 | R25 |
| **S4 Self-improvement eval loop** | `benchmark --learning-delta`, baseline/learning isolation, machine-readable delta report | S2, S3 | R26 (eval) |
| **S5 Long-running harness** | context compaction, `SessionStore`, initializer/progress hooks, `run --session` (may split S5a/S5b) | — | R27 |
| **S6 Release + red-team** | bump 2.0.0, CHANGELOG, upgrade notes, doctor checks, poisoning red-team regression, delta report, **tag v2.0.0** | all | full regression |

**Item→slice map:** brief #1 reflexive write-back = S0+S1+S2 · #2 distillation = S3 ·
#3 harness = S5 · #4 eval = S4.

---

## 5. Cross-cutting invariants (every slice)

- Reflection + distillation + compaction **OFF by default** (opt-in config).
- **Cost-budgeted:** all learning LLM calls (reflect/distill/compaction summary) count
  against the run's V1 reservation (`max_run_usd`/`max_run_tokens`); exhaustion skips
  learning, never overspends.
- **Observable:** OTel spans for reflect/distill/compaction, nested under the run span.
- **Poisoning defense:** recalled memory is DATA never instructions; all written content
  gated (SecretDetection + PromptInjectionScan) like output; provenance tagged;
  soft-deprecate, never hard-delete.
- **Testing:** MockLLM in unit/integration; real LLM only eval tier. `mypy --strict` +
  `ruff` + `pytest` under `--all-extras` green per slice (rule 7b).
- **Process:** one slice = one PR, squash-merge, a lab round each (R22–R27), journals per
  slice. **Branch BEFORE editing** (V1 S7 lesson).
- **New CLAUDE.md rules (added in S1/S3):** learned-content writes go through the
  MemoryAgent gateway; distilled skills follow the knowledge draft→HITL flow.

---

## 6. New config / CLI / env surface

**config.yaml (all optional; defaults preserve prior behavior):**
```yaml
memory:
  enabled: false
  backend: sqlite        # sqlite | null | mock
  path: .lottie/memory.db
  reflect: { enabled: false }
  recall:  { enabled: false }
harness:
  compaction: { enabled: false, max_context_tokens: <int>, keep_recent: <int> }
```

**CLI:** `lottie reflect <agent>`, `lottie distill <agent>`, `lottie distill review`,
`lottie benchmark <agent> --learning-delta`, `lottie run <agent> --session <id>`.

**No new env vars planned** (learning is config-driven per-agent).

---

## 7. Deferred / out of scope (brief item #5)

- **A2A adapter** — wait for the Q3 2026 MCP/A2A joint spec before building.
- **FU-6 msgpack serde** registration for `MeshState`/`StepResult`.
- **Per-node timeouts** in `LangGraphEngine`.
- ChromaDB/vector recall (revisit when a corpus exceeds ~200 files — rule 16).
- Generated-Python distilled skills (template-only in V2; codegen variant is a future
  epic behind the rule-13 pipeline).

---

## 8. Definition of Done (v2.0.0)

- Items 1–4 on main (S0–S5).
- Lab rounds R22–R27 green, **including the memory-poisoning red-team round (R23)**.
- Benchmark shows a **non-negative delta** with learning enabled; the default-on decision
  for write-back is recorded.
- `v2.0.0` tagged.
