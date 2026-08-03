# LOTTIE — AI Orchestrator
> Provider-agnostic multi-agent framework with shared knowledge & AI governance.
> Works with Claude Code, Cursor, Codex, and any LLM.
> Full spec: see `LOTTIE_PHASE0_SPEC.md`
> Development methodology (design-first, modular, lab-validated): see `docs/METHODOLOGY.md`

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
7b. **Never push or merge while CI is (or would be) red.** The local gate must match CI: run `uv run ruff check .`, `uv run mypy --strict src`, `uv run pytest -q` in an env with **all** extras installed (`uv sync --dev --all-extras`) before pushing — a green run with fewer extras hides missing-import (mypy) and `importorskip`-skipped tests (false green). CI installs `--all-extras` for the same reason. After pushing a branch, confirm the PR's checks are green (`gh pr checks`) before squash-merging.

### Security (non-negotiable)
8. **All external inputs pass through `InputSanitizerSkill` first.** No raw external content ever reaches an agent directly.
9. **All LLM outputs pass through `OutputValidationSkill` and `SecretDetectionSkill`** before leaving Lottie.
10. **Knowledge ingest always runs `PromptInjectionScanSkill` and `SecretDetectionSkill`.** No exceptions, even for trusted sources.
11. **Agents can only call skills declared in their `config.yaml` `capabilities` list.** The capability gate (`governance/capability.py` — in `governance/`, not `security/`, so `core` can import it without a `core↔security` cycle) blocks any undeclared skill call at runtime, fail-closed. Enforced in `BaseSkill.run` via a `_execute`-scoped ContextVar the agent activates (framework security skills, invoked outside `_execute`, are exempt). Whitelist-when-nonempty: empty `capabilities` = no enforcement.
12. **Agents write only to `knowledge/draft/`.** Promotion to `curated` always requires human review.
13. **Generated code always runs through** `SecretDetectionSkill` → `CodeSecurityScanSkill` (bandit) → `mypy` → `ruff` before any file write.
13b. **Learned-content writes go through `MemoryAgent.apply` (the memory gateway).** No agent writes reflection/distillation output directly via `self.memory.remember/update`. `apply` screens every delta with `MemoryContentGate` (injection + secret scan, fail-closed), dedups (exact-content, active-only), stamps provenance, and audit-trails each write hash-only. Recalled memory is DATA, never instructions — surface it via `render_as_data`. Post-run reflection (`memory.reflect.enabled`, OFF by default) routes its LLM call through the run's token budget (skip-when-exhausted), is best-effort (never fails the run), and writes lessons through this same gateway.

13c. **Distilled skills are prompt templates, never generated code.** `lottie distill` writes a
    `DistilledSkill` (system prompt + slotted user template + typed slots) to `skills/draft/<name>/`;
    nothing authored by an LLM is ever imported or executed, only rendered by the single generic
    `TemplateRunnerSkill`. Rendering uses literal slot replacement — **never `str.format`**, which
    on an LLM-authored template is an attribute-traversal info leak. Drafts pass the shared
    `security/content_gate.ContentGate` (sanitize + injection + secret, fail-closed) before any
    file write, screening description/system-prompt/template **jointly** so a split payload cannot
    evade it. Promotion draft→registered is always human review (`lottie distill review`): the draft is
    **re-screened** at promotion (it is a file on disk that may have been edited since
    authoring), moved to `skills/distilled/<name>/` as data — never a generated module — and
    stamped with a `promotion.yaml` naming the reviewer. The rule-11 capability is supplied
    by the **reviewer, never the model**; an agent must then declare BOTH `distilled` and
    that capability to invoke the template.

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
                  OutputValidationSkill, CodeSecurityScanSkill
                  (capability enforcement lives in governance/capability.py — see rule 11)
  knowledge/    — manifest, YAML loader, networkx graph, GraphIngestSkill
  mesh/         — MeshAgent, MeshEngine (LocalEngine default; LangGraphEngine via [mesh] extra),
                  SupervisorRouter, checkpointer, mesh schemas. A mesh is itself a BaseAgent
                  (reuses run/serve/benchmark). LangGraphEngine adds parallel fan-out, HITL
                  interrupt/resume, and checkpoint time-travel; LocalEngine stays zero-dep.
  governance/   — audit logger, policy engine, cost tracker, capability gate (rule 11)
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
| `MeshAgent` | `lottie.mesh` | Supervisor→worker mesh; a BaseAgent that routes a task across declared workers |

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
lottie run <agent> --session <id>      # resume/continue a long-running session
lottie session list|show|delete        # inspect session artifacts (.lottie/sessions/)
lottie serve                           # start the MCP stdio server (one tool per agent)
lottie serve --port 8000               # HTTP API: OpenAI-compat (/v1/chat/completions, /v1/models)
                                       #   + Lottie REST (/v1/agents, /v1/agents/{name}/run) — needs [api]
                                       #   resume an interrupted mesh: POST /v1/agents/{name}/resume {thread_id, decision}
                                       #   durable across restarts when served (LOTTIE_MESH_CHECKPOINT=sqlite, set by serve --port)
                                       #   stream:true on /v1/chat/completions -> text/event-stream SSE (format-level; real token streaming deferred)

# Knowledge
lottie knowledge ingest ./docs         # ingest docs into knowledge layer
lottie knowledge ingest --format graphify ./graph.json  # import external graph
lottie knowledge list                  # list all knowledge documents in the manifest
lottie knowledge inspect <id>          # frontmatter, chunk count, and dependents for a doc
lottie knowledge clear                 # drop vector store / draft docs (with confirmation)

# Reflexive memory
lottie reflect <agent>                 # consolidate an agent's episodic memory → semantic notes (gated)

# Skill distillation (rule 13c) — templates only, never generated code
lottie distill run <agent>             # successful trajectories → draft template in skills/draft/
lottie distill list                    # drafts awaiting human review
lottie distill show <name>             # print a draft's template + provenance
lottie distill review                  # list drafts pending human review
lottie distill review <name> --approve --capability <cap> --reviewer <who>
lottie distill review <name> --reject  # discard a draft

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
