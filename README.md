# LOTTIE — AI Orchestrator

![Status](https://img.shields.io/badge/status-active-success) [![codecov](https://codecov.io/github/cdiaz19/lottie-orchestrator/graph/badge.svg?token=WX4QZOEEJG)](https://codecov.io/github/cdiaz19/lottie-orchestrator)

Provider-agnostic multi-agent AI orchestration framework with shared knowledge and AI governance.

> Works with Claude Code, Cursor, Codex, and any LLM. Swap Claude for GPT-4o with a single config change — no code changes, ever.

<img width="1456" height="720" alt="Gemini_Generated_Image_sopbzlsopbzlsopb" src="https://github.com/user-attachments/assets/26445481-94e2-42fe-b185-79f68c39a398" />

**`v3.3.0`** — recorded mesh plans and deterministic replay. Full history in [`CHANGELOG.md`](CHANGELOG.md).

---

## Why Lottie

- **Provider-agnostic.** Agent and skill code never imports an LLM SDK. Every call goes through `lottie.llm.LLMProvider`; the provider is chosen by config (`litellm` under the hood). Mock it in tests, swap it in prod.
- **Typed end to end.** Every input/output crossing an agent or skill boundary is a Pydantic v2 model — no raw dicts or strings. `mypy --strict` across the whole tree.
- **Secure by construction.** External input is sanitized, LLM output is validated + secret-scanned, knowledge ingest is prompt-injection scanned, agents can only call the skills they declare, and generated code passes a SecretDetection → bandit → mypy → ruff gate before any file write.
- **Knowledge as a first-class layer.** YAML-frontmatter docs are the source of truth; a networkx graph is the query layer (impact / audit / cycles); ChromaDB is added only when the corpus is large.
- **Governed.** Per-run latency / tokens / cost instrumentation, audit logging, and policy hooks are built into `BaseAgent`.

## Architecture at a glance

```
src/lottie/
  core/         BaseAgent, BaseSkill, registry (auto-instrumented run loop)
  llm/          LLMProvider, MockLLMProvider, litellm adapter
  cli/          Typer CLI (init, create, run, list, inspect, benchmark, memory, mesh, serve)
  security/     InputSanitizer, SecretDetection, PromptInjectionScan,
                OutputValidation, CapabilityEnforcer, CodeSecurityScan
  knowledge/    YAML manifest, networkx graph, embeddings, vector store, ingest
  mesh/         MeshAgent, MeshEngine ABC (LocalEngine default; LangGraphEngine
                via [mesh] extra), SupervisorRouter, checkpointer
  serve/        transport-agnostic AgentService + SecurityGate; MCP stdio + HTTP (OpenAI-compat + REST) transports
  governance/   audit logger, policy engine, cost tracker, OpenTelemetry tracer
```

Key abstractions: `LLMProvider` (swap providers), `BaseAgent` / `BaseSkill` (typed, auto-benchmarked), `SecurityGate` (fail-closed input/output chokepoint on the serve path), `MeshAgent` (supervisor→worker mesh; itself a `BaseAgent`), `AgentService` (transport-agnostic serving core), `KnowledgeManifest` (YAML + graph).

## Quickstart

```bash
lottie init my-project              # scaffold a project
lottie create agent digest          # scaffold a typed agent (all required files)
lottie run digest --input '{"query": "multi-agent AI systems"}'
lottie benchmark agent digest       # record latency, cost, accuracy
```

### Knowledge & memory graph

```bash
lottie knowledge ingest ./docs      # scan → chunk → embed → store (injection/secret gated)
lottie knowledge list               # list documents in the manifest
lottie memory impact <file>         # what breaks if this is deprecated?
lottie memory audit                 # cycles, orphans, stale deps
```

### Self-learning memory

Every learned-content write goes through the `MemoryAgent` gateway, which injection- and
secret-screens it, stamps provenance, and audit-trails it hash-only. All of it is opt-in
per agent, in `config.yaml`:

```yaml
memory:
  enabled: true
  backend: sqlite              # sqlite | null | mock
  path: .lottie/memory.db
  namespace: null              # null → the agent's own name
  recall:
    enabled: false             # inject semantic notes as DATA, never as instructions
    limit: 5
  reflect:
    enabled: false             # post-run: distil the run into lessons (spends tokens)
  trajectory:
    enabled: false             # append each run to episodic memory (spends NO tokens)
    max_chars: 4000            # per-field bound on stored task/outcome text
```

Trajectory persistence is what gives `lottie reflect` and skill distillation a corpus to
read — without it the episodic tier stays empty. Records land in the EPISODIC tier, which
recall-as-data never reads, so a raw trajectory can never reach a prompt.

```bash
lottie reflect digest               # consolidate episodic runs → durable semantic notes
lottie distill run digest           # successful runs → a reusable draft skill template
lottie distill list                 # drafts awaiting human review
lottie distill review               # what is pending, and what was promoted
lottie distill review digest_distilled --approve --capability digestion --reviewer ana
```

Distillation turns repeated successful runs into a **parameterized prompt template**, never
generated Python — nothing authored by the model is imported or executed, only rendered by the
single generic `TemplateRunnerSkill`. Drafts land in `skills/draft/<name>/` and pass the same
injection/secret screen as memory writes before touching disk. Promotion to a registered skill
is always a human decision: `lottie distill review --approve` re-screens the draft, moves it
to `skills/distilled/<name>/`, and records who approved it under which capability. An agent
must declare both `distilled` and that capability to invoke it.

### Replay a mesh run

A mesh routes **dynamically** — the supervisor decides each step from what already
happened — which makes a multi-agent flow non-deterministic and awkward to test or debug.
Every completed run now records the decisions it actually made:

```bash
lottie plan list assistant           # runs with a recorded plan
lottie plan show assistant <thread>  # the routing decisions, step by step
```

A recorded plan can be replayed with **zero supervisor calls**, which turns a
non-deterministic flow into a repeatable one: regression tests over multi-agent behaviour,
and debugging a failure without paying for routing again.

The plan stores a **hash** of the task, never its text — the same discipline that keeps
raw content out of the audit ledger. Replaying against a mesh that no longer declares a
recorded worker fails loudly rather than silently skipping it.

### Provider fallback

```yaml
# lottie.yaml
providers:
  default: anthropic/claude-sonnet-4-6
  fallback: openai/gpt-4o
```

When the primary provider fails **transiently** — rate limit, timeout, 5xx, connection
error — the run continues on the fallback. The audit record names the model that
*actually served* it, and a warning fires at the moment it happens, so a fallback is never
silent.

It deliberately does **not** fall back on a content-policy refusal. Shopping a refused
request to a second model would launder a provider's safety decision through a framework
that advertises fail-closed gates. Bad requests and auth errors also fail fast: they fail
identically on the fallback, so retrying only doubles the spend.

With no `fallback` configured, nothing is wrapped and nothing changes.

### Inspect what wraps a run

Every agent run is wrapped by an ordered chain of runtime modules — security gates, policy,
budget, capability, memory. The chain is otherwise invisible, and inference is exactly how
a disabled gate goes unnoticed:

```bash
lottie modules             # every agent's mounted chain, in execution order
lottie modules digest      # one agent
```

A module can be switched off per agent. Built-in modules keep their existing config keys
(`budget_usd`, `capabilities`, `memory.*`); this block is only for turning one **off**:

```yaml
modules:
  recall: { enabled: false }
```

`lottie doctor` flags an unknown module name (a typo there does nothing, which is the
dangerous kind of nothing) and warns loudly when a **fail-closed** module —
`security_input`, `security_output`, `policy`, `capability` — is disabled.

### Long runs: context compaction

```yaml
harness:
  compaction:
    enabled: false          # OFF by default; spends tokens (one LLM call per compaction)
    max_context_tokens: 8000
    keep_recent: 6          # most recent turns kept verbatim
```

When a run approaches its context window, older turns are summarised into a single
`[compacted history]` message and the recent ones are kept verbatim. System messages are
**pinned** — they carry the recall-as-data block, which is a security contract, not a
nicety, so compaction never removes it.

Compaction is best-effort: if summarising fails the full context is sent instead, because
the provider's own context error is a clearer signal than a summariser outage disguised as
a task failure. A token-cap trip is *not* swallowed — that is the run's decision to make.

### Long runs: sessions

A session lets a task survive beyond one process — the agent writes incremental progress,
the process exits, and a later run picks up where it left off.

```bash
lottie run digest --input '{"query": "..."}' --session nightly
lottie session list                 # id, agent, run count, progress keys
lottie session show nightly         # full state as JSON
lottie session delete nightly
```

An agent opts in by reading `self.session_progress` and calling `self.save_progress(...)`:

```python
raw = self.session_progress.get("step", 0)
step = raw if isinstance(raw, int) else 0
self.save_progress(step=step + 1)
```

Progress is saved on **every** `save_progress` call, not once at the end, so a run that
dies halfway still leaves behind what it achieved. Run history is recorded hash-only, the
same discipline as the audit ledger: it shows *that* the session progressed and what it
cost, never the content.

Progress is screened on write like any memory write — it round-trips into a future run, so
an agent storing raw model output would otherwise have a way to smuggle instructions across
process boundaries. On the way back in it is **data, never instructions**.

### Does learning actually help?

```bash
lottie benchmark agent digest --learning-delta
```

Runs the eval suite twice on the same provider — recall **off**, then **on** — and reports the
per-metric difference plus a verdict (`improved` / `neutral` / `regressed`, judged on accuracy).
A machine-readable report lands in `.lottie/benchmarks/<agent>-learning-delta.json`; it is the
evidence behind any decision to turn learning on by default.

Both arms disable every memory **write**, not just the baseline. A benchmark that wrote
trajectories or lessons would mutate the corpus it measures, so a second run would silently
report different numbers. The report also states how many notes were recalled — a `neutral`
verdict over an empty store means the experiment never ran, which is not the same as learning
not helping.

> Trajectories store raw task and outcome text, unlike the audit ledger which stores only
> hashes. They are gated on write and size-bounded, but a project handling sensitive input
> should leave `trajectory.enabled` off.

### Run a multi-agent mesh

A mesh is itself an agent: a supervisor routes each task across declared workers, with token/cost rolled up. The default `LocalEngine` is zero-dep; the optional `[mesh]` extra swaps in a `LangGraphEngine` for parallel fan-out, human-in-the-loop interrupt/resume, and checkpoint time-travel.

```bash
lottie run assistant --input '{"query": "research multi-agent systems"}'        # mesh runs like any agent
lottie mesh resume --agent assistant --thread-id <id> --decision approve         # continue a paused HITL run
lottie mesh history --agent assistant --thread-id <id>                           # report available checkpoint history
```

> HITL resume/history need the `[mesh]` extra. The default in-memory checkpointer is process-local; durable cross-process resume uses the sqlite checkpointer (on the roadmap).

### Serve agents — MCP stdio or HTTP

Expose every agent as a typed MCP tool to any host (Claude Code, Cursor, Codex) — the host's LLM sees `digest(query: str)` and calls it directly:

```bash
pip install "lottie-orchestrator[serve]"   # optional MCP transport extra
lottie serve                               # MCP stdio server (one tool per agent)
```

Or serve an HTTP API on a port — two surfaces over one server, sharing the same fail-closed `SecurityGate` + audit/policy/cost path (no second gate):

```bash
pip install "lottie-orchestrator[api]"     # optional HTTP transport extra (Starlette + uvicorn)
lottie serve --port 8000
```

- **OpenAI-compatible** — `POST /v1/chat/completions` + `GET /v1/models`. Point any OpenAI client's base URL here; an agent opts in by declaring a `chat: {input_field, output_field}` block in its `config.yaml`.
- **Lottie REST** — `GET /v1/agents`, `GET /v1/agents/{name}` (Input JSON schema), `POST /v1/agents/{name}/run` (the agent's typed Input → the full `RunResult`). Every agent is reachable, no opt-in. A mesh that hits a human-in-the-loop gate returns `status:"interrupted"` + a `thread_id`; resume it with `POST /v1/agents/{name}/resume` (`{thread_id, decision}`) — **durable across restarts/workers** when served (the engine checkpoints to sqlite; `LOTTIE_MESH_CHECKPOINT=sqlite` is set by `serve --port`).

`stream:true` on the chat endpoint returns a `text/event-stream` (SSE) response — format-level: the agent runs fully, then streams its output as OpenAI `chat.completion.chunk` events (real token-by-token streaming is still deferred). A dead/over-budget run fails closed (governance is inherited from the run chokepoint).

## Testing

| Layer | What | LLM? | Command |
|---|---|---|---|
| Unit | Skills | ✗ Mock | `pytest skills/` |
| Integration | Agents (MockLLM) | Mocked | `pytest agents/` |
| Contract | Pydantic schemas | ✗ None | `pytest tests/contracts/` |
| Eval | Quality benchmarks | ✓ Real | `lottie benchmark agent` |

`mypy --strict src` + `ruff check .` gate every change. Unit/integration tests never call a real LLM.

## Roadmap

| Tag | Phase | Ships | Status |
|---|---|---|---|
| `v0.1.0` | 0 — Foundations | BaseAgent/Skill, CLI, generators, MockLLM | ✅ |
| `v0.2.0` | 1 — Knowledge Core | ChromaDB, RAG pipeline, knowledge graph | ✅ |
| `v0.3.0` | 2 — Agent Mesh | Supervisor→worker mesh, conditional routing, typed state (parallel/HITL/time-travel → Phase 3) | ✅ |
| `v0.4.0` | 3 — Mesh Hardening | LangGraph backend, parallel fork/join, human-in-the-loop, time-travel (opt-in `[mesh]` extra) | ✅ |
| _later_ | Governance | immutable audit trail (`lottie audit`) + capability policy engine (allow/deny/escalate) + per-agent cost budgets (fail-closed circuit-breaker) + OpenTelemetry tracing (opt-in `[otel]`, fail-open, no-op default) | ✅ audit + policy + cost + otel |
| `v0.5.0` | 4 — Integration | MCP stdio + HTTP API (OpenAI-compat `/v1/chat/completions`, Lottie REST `/v1/agents`, durable mesh resume, SSE + real token streaming) | ✅ MCP + OpenAI-compat + REST + resume + streaming |
| `v1.0.0` | V1 — Hardening | "complete, secured, documented": rule-11 capability enforcement, BaseAgent/CLI security gate, per-run token cap + TOCTOU-safe atomic cost reservation, HTTP auth/rate-limit/pagination, HITL edited_input, agentic hygiene (`max_turns` + `_verify`) | ✅ |
| `v2.0.0` | V2 — Self-learning & harness | persistent memory + fail-closed write gateway, recall-as-data, reflexive write-back, episodic trajectories, skill distillation to prompt templates (never codegen) with human promotion, learning-delta benchmark, context compaction, resumable sessions | ✅ |
| `v3.0.0` | V3 — Runtime kernel | execution kernel (abort-capable middleware chain + fail-open event stream), one execution path for `run`/`run_stream`, modules owned by their subsystems, auditing as a subscriber, `lottie modules` + config-driven module control | ✅ |
| `v3.1.0` | E4 — Context Compiler | ordered, budgeted, provenance-carrying message assembly; pinning by source; reflection as a module | ✅ |
| `v3.2.0` | E5 — Provider Router | `providers.fallback` honoured; transient-only fallback that never launders a policy refusal; knowledge as a droppable context source | ✅ |
| `v3.3.0` | E6 — Execution Planner | recorded plans + deterministic replay (zero supervisor calls); `lottie plan` | ✅ |
| _next_ | V3.4 | Plugin SDK — public extension API, opt-in loading by explicit name | ◻ |

It's verified in the open — see the [lottie-lab](https://github.com/cdiaz19/lottie-lab) round-by-round test harness.

## Coverage

[![codecov tree](https://codecov.io/github/cdiaz19/lottie-orchestrator/graphs/tree.svg?token=WX4QZOEEJG)](https://codecov.io/github/cdiaz19/lottie-orchestrator)

---

Full architecture spec: [`LOTTIE_PHASE0_SPEC.md`](./LOTTIE_PHASE0_SPEC.md)
