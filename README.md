# LOTTIE — AI Orchestrator

![Status](https://img.shields.io/badge/status-active-success) [![codecov](https://codecov.io/github/cdiaz19/lottie-orchestrator/graph/badge.svg?token=WX4QZOEEJG)](https://codecov.io/github/cdiaz19/lottie-orchestrator)

Provider-agnostic multi-agent AI orchestration framework with shared knowledge and AI governance.

> Works with Claude Code, Cursor, Codex, and any LLM. Swap Claude for GPT-4o with a single config change — no code changes, ever.

**Status:** Phase 3 — Mesh Hardening shipped (`v0.4.0`): a `LangGraphEngine` (optional `[mesh]` extra) adds parallel fan-out, human-in-the-loop interrupt/resume, and checkpoint time-travel behind the engine ABC; the hand-rolled `LocalEngine` stays the zero-dep default. Phase 2 (Agent Mesh core, `v0.3.0`), Phases 0–1 (foundations, knowledge core), and the Phase-4 HTTP transports — MCP stdio plus an `lottie serve --port` HTTP API (OpenAI-compat `/v1/chat/completions` + Lottie REST `/v1/agents`, opt-in `[api]` extra) — all shipped. Governance has begun landing on `main`: a fail-closed serve-path `SecurityGate`, an immutable per-run audit trail (`lottie audit`), a declarative capability policy engine (allow/deny/escalate), per-agent cost budgets (a fail-closed cumulative circuit-breaker), and OpenTelemetry tracing (opt-in `[otel]` extra, one fail-open span per run, no-op by default).

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
- **Lottie REST** — `GET /v1/agents`, `GET /v1/agents/{name}` (Input JSON schema), `POST /v1/agents/{name}/run` (the agent's typed Input → the full `RunResult`). Every agent is reachable, no opt-in.

Non-streaming for now; a dead/over-budget run fails closed (governance is inherited from the run chokepoint).

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
| `v0.5.0` | 4 — Integration | MCP stdio + HTTP API (OpenAI-compat `/v1/chat/completions`, Lottie REST `/v1/agents`); resume-over-REST + streaming next | ✅ MCP + OpenAI-compat + REST |
| `v1.0.0` | 5 — Public SDK | docs site, plugin system, demos | ◻ |

It's verified in the open — see the [lottie-lab](https://github.com/cdiaz19/lottie-lab) round-by-round test harness.

## Coverage

[![codecov tree](https://codecov.io/github/cdiaz19/lottie-orchestrator/graphs/tree.svg?token=WX4QZOEEJG)](https://codecov.io/github/cdiaz19/lottie-orchestrator)

---

Full architecture spec: [`LOTTIE_PHASE0_SPEC.md`](./LOTTIE_PHASE0_SPEC.md)
