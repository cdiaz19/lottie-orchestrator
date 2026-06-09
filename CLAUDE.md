# LOTTIE — AI Orchestrator
> Provider-agnostic multi-agent framework with shared knowledge & AI governance.
> Works with Claude Code, Cursor, Codex, and any LLM.
> Full spec: see `LOTTIE_PHASE0_SPEC.md`

---

## Rules — read before writing any code

### Core
1. **Never import an LLM SDK directly.** All LLM calls go through `lottie.llm.LLMProvider`. Never use `anthropic`, `openai`, or any provider SDK in agent or skill code.
2. **All inputs/outputs are Pydantic v2 models.** Defined in `schema.py`. No raw dicts or strings crossing agent/skill boundaries.
3. **Every agent needs `AGENT.md`. Every skill needs `SKILL.md`.** Write the doc before the code.
4. **Use `lottie create agent <name>` or `lottie create skill <name>`** to scaffold. Never create agent/skill files manually.
5. **Unit tests must not call real LLMs.** Use `MockLLMProvider` from `lottie.llm`. Real LLM calls only in eval tests.
6. **Every file must pass `mypy --strict`.** No `Any` types without explicit justification.
7. **Commit convention:** `feat:`, `fix:`, `docs:`, `chore:`, `test:` — conventional commits only.

### Security (non-negotiable)
8. **All external inputs pass through `InputSanitizerSkill` first.** No raw external content ever reaches an agent directly.
9. **All LLM outputs pass through `OutputValidationSkill` and `SecretDetectionSkill`** before leaving Lottie.
10. **Knowledge ingest always runs `PromptInjectionScanSkill` and `SecretDetectionSkill`.** No exceptions, even for trusted sources.
11. **Agents can only call skills declared in their `config.yaml` `capabilities` list.** The `CapabilityEnforcerSkill` blocks anything else at runtime.
12. **Agents write only to `knowledge/draft/`.** Promotion to `curated` always requires human review.
13. **Generated code always runs through** `SecretDetectionSkill` → `CodeSecurityScanSkill` (bandit) → `mypy` → `ruff` before any file write.

### Knowledge
14. **YAML frontmatter on every knowledge file** — `id`, `layer`, `scope`, `tags`, `status`, `last_verified`, `depends_on`.
15. **Files are the source of truth. networkx graph is the query layer** built at runtime from the manifest. Do not use ChromaDB for layers 0–2.
16. **Structured retrieval (`yq` filter) before semantic search.** Add ChromaDB only when corpus exceeds ~200 files.

---

## Project structure

```
src/lottie/
  core/         — BaseAgent, BaseSkill, registry
  llm/          — LLMProvider, MockLLMProvider, litellm adapter
  cli/          — Typer CLI (lottie run, create, benchmark, memory, serve)
  security/     — InputSanitizerSkill, SecretDetectionSkill, PromptInjectionScanSkill,
                  OutputValidationSkill, CapabilityEnforcerSkill, CodeSecurityScanSkill
  knowledge/    — manifest, YAML loader, networkx graph, GraphIngestSkill
  governance/   — audit logger, policy engine, cost tracker
agents/         — user-defined agents (each a self-contained module)
skills/         — user-defined skills (stateless, deterministic)
knowledge/      — raw docs (YAML frontmatter + content)
  global/       — always injected, <500 tokens target
  platform/     — per-platform context
  project/      — per-project context
  memory/       — task-relevant, tag-matched
  draft/        — AI-generated, awaiting human review
policies/       — YAML governance rules (allow/deny/escalate)
tests/
  contracts/    — Pydantic schema validation tests
  e2e/          — full pipeline tests (Phase 2+)
.lottie/        — runtime (gitignored): chroma/, audit.db, benchmarks/
.private-journey/ — personal context (gitignored, read by Claude Code if present)
```

---

## Key abstractions

| Class | Location | Purpose |
|---|---|---|
| `LLMProvider` | `lottie.llm` | Abstract LLM interface — swap providers via config |
| `MockLLMProvider` | `lottie.llm` | For tests — returns pre-defined responses |
| `BaseAgent` | `lottie.core` | All agents extend this — auto-instruments latency/tokens/cost |
| `BaseSkill` | `lottie.core` | All skills extend this — typed input/output, auto-benchmarked |
| `SecurityGate` | `lottie.security` | Input + output security checkpoint — wraps every agent run |
| `KnowledgeManifest` | `lottie.knowledge` | YAML manifest + networkx graph builder |

---

## CLI commands

```bash
# Project
lottie init <name>                     # scaffold new project
lottie status                          # show agents, skills, knowledge, providers
lottie doctor                          # check environment health

# Generators
lottie create agent <name>             # scaffold agent (all required files)
lottie create skill <name>             # scaffold skill (all required files)
lottie create agent <name> --from-desc "..."  # AI-powered generator

# Registry
lottie list agents                     # list agents with provider
lottie list skills                     # list skills with input/output types
lottie inspect agent <name>            # config, schema, system prompt
lottie inspect skill <name>            # schema, SKILL.md presence

# Running
lottie run <agent>                     # run an agent
lottie run <agent> --provider openai   # override provider for this run
lottie serve --port 8080               # start MCP + OpenAI-compat + REST

# Knowledge
lottie knowledge ingest ./docs         # ingest docs into knowledge layer
lottie knowledge ingest --format graphify ./graph.json  # import external graph
lottie knowledge list                  # list all knowledge documents in the manifest
lottie knowledge inspect <id>          # frontmatter, chunk count, and dependents for a doc
lottie knowledge clear                 # drop vector store / draft docs (with confirmation)

# Memory graph
lottie memory graph                    # visualize dependency graph
lottie memory impact <file>            # what breaks if this is deprecated?
lottie memory audit                    # find cycles, orphans, stale deps (90d)
lottie memory review                   # surface drafts for human review

# Benchmarking & audit
lottie benchmark agent <name>          # run eval suite, record all metrics
lottie benchmark agent <name> --compare  # compare across providers
lottie report performance              # trend charts
lottie audit --agent <name>            # query immutable audit log
```

---

## Testing rules

| Layer | What | LLM? | Command |
|---|---|---|---|
| Unit | Skills only | ✗ None | `pytest skills/` |
| Integration | Agents (MockLLM) | Mocked | `pytest agents/` |
| Contract | Pydantic schemas | ✗ None | `pytest tests/contracts/` |
| Eval | Quality benchmarks | ✓ Real | `lottie benchmark agent` |
| E2E | Full pipeline | ✓ Real | `pytest tests/e2e/` (on tag only) |

---

## Private context

If `.private-journey/context.md` exists, read it for additional project context before responding.
