# LOTTIE — AI Orchestrator

[![codecov](https://codecov.io/github/cdiaz19/lottie-orchestrator/graph/badge.svg?token=WX4QZOEEJG)](https://codecov.io/github/cdiaz19/lottie-orchestrator)

Provider-agnostic multi-agent AI orchestration framework with shared knowledge and AI governance.

> Works with Claude Code, Cursor, Codex, and any LLM. Swap Claude for GPT-4o with a single config change — no code changes, ever.

**Status:** Phase 1 — Knowledge Core shipped (`v0.2.0`). Phase 4 (Integration Layer) in progress: first transport — MCP stdio server — on branch `feat/mcp-stdio`.

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
  cli/          Typer CLI (init, create, run, list, inspect, benchmark, memory, serve)
  security/     InputSanitizer, SecretDetection, PromptInjectionScan,
                OutputValidation, CapabilityEnforcer, CodeSecurityScan
  knowledge/    YAML manifest, networkx graph, embeddings, vector store, ingest
  serve/        transport-agnostic AgentService + SecurityGate; MCP stdio transport
  governance/   audit logger, policy engine, cost tracker
```

Key abstractions: `LLMProvider` (swap providers), `BaseAgent` / `BaseSkill` (typed, auto-benchmarked), `SecurityGate` (input/output chokepoint on every run), `AgentService` (transport-agnostic serving core), `KnowledgeManifest` (YAML + graph).

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

### Serve agents over MCP

Expose every agent as a typed MCP tool to any host (Claude Code, Cursor, Codex) — the host's LLM sees `digest(query: str)` and calls it directly:

```bash
pip install "lottie-orchestrator[serve]"   # optional MCP transport extra
lottie serve                               # MCP stdio server (one tool per agent)
```

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
| `v0.3.0` | 2 — Agent Mesh | LangGraph engine, supervisor, parallel runner | ◻ |
| `v0.4.0` | 3 — Governance | audit trail, policy engine, OpenTelemetry | ◻ |
| `v0.5.0` | 4 — Integration | MCP server, OpenAI-compat API, REST | 🚧 MCP stdio |
| `v1.0.0` | 5 — Public SDK | docs site, plugin system, demos | ◻ |

It's verified in the open — see the [lottie-lab](https://github.com/cdiaz19/lottie-lab) round-by-round test harness.

## Coverage

[![codecov tree](https://codecov.io/github/cdiaz19/lottie-orchestrator/graphs/tree.svg?token=WX4QZOEEJG)](https://codecov.io/github/cdiaz19/lottie-orchestrator)

---

Full architecture spec: [`LOTTIE_PHASE0_SPEC.md`](./LOTTIE_PHASE0_SPEC.md)
