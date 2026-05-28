# 🐶 LOTTIE — Phase 0 Specification
> **Foundations** · Weeks 1–4 · Spec-first document

---

## 1. Mental Model: Agent vs Skill

### Agent — THINKS · DECIDES · ACTS
An Agent is an LLM-backed, stateful, role-driven unit that reasons and decides.

| Property | Description |
|---|---|
| LLM-backed | Has an assigned provider (Claude, GPT-4o, local). All reasoning goes through it. |
| Stateful | Maintains memory across steps. Reads/writes to the shared knowledge layer. |
| Role-driven | Has a system prompt defining persona, goals, and constraints. |
| Tool-user | Calls Skills as tools. The agent decides *when* and *which* skill to invoke. |
| Governed | Every decision is logged. Policies checked before every action. |

**Examples:** `ResearcherAgent`, `ReviewerAgent`, `PlannerAgent`, `IncidentResponderAgent`

---

### Skill — EXECUTES · DETERMINISTIC · REUSABLE
A Skill is a stateless, deterministic capability with a clear typed contract.

| Property | Description |
|---|---|
| Deterministic | No LLM required. Same input → predictable output. Easy to unit test. |
| Stateless | Does not hold memory. Receives typed input, returns typed output. |
| Composable | Registered as a tool on any Agent. Multiple agents share one skill. |
| Benchmarkable | Latency, cost, and accuracy tracked independently. |
| May use LLM internally | e.g. `SummarizeSkill` wraps an LLM call, but callers don't care. |

**Examples:** `WebSearchSkill`, `CodeParserSkill`, `DocumentIngestSkill`, `BenchmarkSkill`

---

## 2. The Golden Rules — "The Lottie Way"

1. **Agents make decisions. Skills execute them.** If it involves reasoning or choosing, it's an agent. If it's a capability with a clear contract, it's a skill.
2. **Never import an LLM SDK directly.** All LLM calls go through `lottie.llm.LLMProvider`. Swap Claude for GPT-4o with one config change.
3. **Every Agent and Skill has an AGENT.md / SKILL.md.** If it isn't documented, it doesn't exist. The docs define the contract.
4. **Every input/output is a typed Pydantic model.** No raw dicts or strings crossing agent/skill boundaries.
5. **Everything is benchmarkable from day one.** Base classes emit timing and token metrics automatically — no extra code needed.
6. **One command to scaffold, one to run, one to benchmark.** `lottie create`, `lottie run`, `lottie benchmark` — consistency everywhere.

---

## 3. CLI Commands

### Project
| Command | Description |
|---|---|
| `lottie init <project-name>` | Scaffold a new Lottie project — folder structure, lottie.yaml, LOTTIE.md, hello-world agent |
| `lottie status` | Show registered agents, skills, knowledge base size, provider config, active policies |
| `lottie doctor` | Check environment health — API keys set, providers reachable, dependencies installed |

### Generators — The Self-Extending Core
| Command | Description |
|---|---|
| `lottie create agent <name>` | Scaffold a complete agent from template (all required files) |
| `lottie create skill <name>` | Scaffold a complete skill from template (all required files) |
| `lottie create agent <name> --from-desc "..."` | AI-powered: generate a full working agent from a natural-language description |
| `lottie create skill <name> --from-desc "..."` | AI-powered: generate a full skill implementation from description |

### Running
| Command | Description |
|---|---|
| `lottie run <agent>` | Run an agent interactively, streaming output to terminal |
| `lottie run <agent> --input '{"key":"val"}'` | Run with a JSON input payload |
| `lottie run <agent> --provider openai` | Override the LLM provider for this run (for cost comparison) |
| `lottie serve --port 8080` | Start all integration endpoints: MCP, OpenAI-compat API, REST, WebSocket |

### Registry
| Command | Description |
|---|---|
| `lottie list agents` | List all registered agents with provider, version, last-run, benchmark score |
| `lottie list skills` | List all registered skills with input/output types and benchmark score |
| `lottie inspect agent <name>` | Full config, schema, system prompt, tools, policies, performance history |

### Benchmarking & Audit
| Command | Description |
|---|---|
| `lottie benchmark agent <name>` | Run eval suite, record latency/tokens/cost/accuracy |
| `lottie benchmark agent <name> --compare` | Same eval across all providers — cost/quality side-by-side |
| `lottie report performance` | Trend charts: latency P50/P95, cost per run, error rate |
| `lottie audit --agent <name>` | Query the immutable audit log, filter by agent/provider/date/policy |

---

## 4. Project File Structure

```
my-project/
│
│   # Project config & docs
├── lottie.yaml             # providers, policies, registry paths
├── LOTTIE.md               # project docs (read by all AI tools automatically)
├── pyproject.toml          # deps, typing, linters
│
│   # Agents — each is a self-contained module
├── agents/
│   ├── __init__.py         # auto-discovers and registers all agents
│   └── researcher/
│       ├── AGENT.md        # contract, role, examples ← REQUIRED
│       ├── agent.py        # class ResearcherAgent(BaseAgent)
│       ├── schema.py       # ResearchInput / ResearchOutput (Pydantic v2)
│       ├── config.yaml     # provider, params, tools, policies
│       ├── prompts.py      # system prompt, few-shot examples
│       └── tests/
│           └── test_researcher.py
│
│   # Skills — stateless, reusable capabilities
├── skills/
│   ├── __init__.py         # auto-discovers and registers all skills
│   └── web_search/
│       ├── SKILL.md        # what it does, inputs, outputs ← REQUIRED
│       ├── skill.py        # class WebSearchSkill(BaseSkill)
│       ├── schema.py       # SearchInput / SearchOutput (Pydantic v2)
│       └── tests/
│           └── test_web_search.py
│
│   # Shared knowledge, policies, and runtime
├── knowledge/              # raw docs ingested via `lottie knowledge ingest`
├── policies/
│   └── base.yaml           # allow/deny/escalate rules
└── .lottie/                # runtime — gitignored
    ├── chroma/             # vector store
    ├── audit.db            # immutable audit log (SQLite)
    └── benchmarks/         # performance records (JSONL)
```

---

## 5. File Standards

### AGENT.md (required for every agent)
```markdown
# <AgentName>

## Role
One sentence describing what this agent does.

## Input
| Field | Type | Description |
|---|---|---|
| query | str | The user's query |

## Output
| Field | Type | Description |
|---|---|---|
| result | str | The agent's response |
| sources | list[str] | Sources cited |

## Provider
Default: anthropic/claude-sonnet-4-6

## Tools (Skills used)
- WebSearchSkill
- DocumentIngestSkill

## Policies
- base
- no-pii

## Examples
### Example 1
Input: ...
Output: ...
```

### SKILL.md (required for every skill)
```markdown
# <SkillName>

## What it does
One sentence description.

## Input
| Field | Type | Required | Description |
|---|---|---|---|

## Output
| Field | Type | Description |
|---|---|---|

## Side effects
None / list any.

## Examples
### Example 1
Input: ...
Output: ...
```

### config.yaml (agent)
```yaml
provider: anthropic/claude-sonnet-4-6
model_params:
  temperature: 0.3
  max_tokens: 2048
tools:
  - WebSearchSkill
  - DocumentIngestSkill
policies:
  - base
  - no-pii
memory:
  enabled: true
  namespace: researcher
```

---

## 6. Generator: `lottie create agent/skill`

### How it works
1. **User input** — name + optional `--from-desc "..."`
2. **ScaffolderAgent** — Lottie's internal meta-agent decides what to generate
3. **Template render** — Jinja2 templates via `TemplateRendererSkill`
4. **Validate** — `SchemaValidatorSkill` checks all generated files pass mypy and schema rules
5. **Register** — auto-adds to registry and updates `LOTTIE.md`

### ScaffolderAgent system prompt (core rules)
```
You are the Lottie ScaffolderAgent. Generate production-ready agent and skill
code that follows the Lottie standards exactly.

Rules you must never break:
1. All LLM calls use lottie.llm.LLMProvider — never openai or anthropic directly
2. All inputs/outputs are Pydantic v2 models in schema.py
3. AGENT.md / SKILL.md must be written before any code
4. Every generated file must pass mypy --strict
5. Generate at least 3 test cases: happy path, edge case, error
6. config.yaml must reference policies by name, never hardcode rules in code
```

### Files generated by `lottie create agent <name>`
| File | Required | Description |
|---|---|---|
| AGENT.md | ✅ | Role, contract, examples |
| agent.py | ✅ | extends BaseAgent |
| schema.py | ✅ | Input/Output Pydantic models |
| config.yaml | ✅ | provider, tools, policies |
| prompts.py | ✅ | system prompt template |
| tests/test_\<name\>.py | ✅ | 3 baseline test cases |
| evals/eval_\<name\>.py | optional | benchmark eval suite |

### Files generated by `lottie create skill <name>`
| File | Required | Description |
|---|---|---|
| SKILL.md | ✅ | What it does, inputs, outputs |
| skill.py | ✅ | extends BaseSkill |
| schema.py | ✅ | Input/Output Pydantic models |
| tests/test_\<name\>.py | ✅ | Unit tests (deterministic) |
| evals/eval_\<name\>.py | optional | benchmark eval suite |

---

## 7. Performance Benchmarking

### Metrics tracked automatically by BaseAgent / BaseSkill
| Metric | Type | Description |
|---|---|---|
| agent_name | str | name of the agent or skill |
| provider | str | e.g. "anthropic/claude-sonnet-4-6" |
| timestamp | datetime | when the benchmark ran |
| latency_p50_ms | float | median latency in milliseconds |
| latency_p95_ms | float | 95th percentile latency |
| input_tokens | int | avg input tokens per run |
| output_tokens | int | avg output tokens per run |
| cost_usd | float | avg cost per run in USD |
| accuracy_pct | float | % of eval cases passed |
| retry_rate | float | fraction of runs that retried at least once |
| version | str | git commit hash — ties results to code |

### Auto-instrumentation
- `BaseAgent` and `BaseSkill` wrap every `run()` call in a context manager recording start time, end time, and token usage — **zero extra code in your agents**.
- Results are appended to `.lottie/benchmarks/<agent>.jsonl` on every run in dev mode. In production they stream to OpenTelemetry.
- The `--compare` flag reruns the same eval suite per provider and computes means — exact cost/quality tradeoff before committing to a provider.

---

## 8. Private Journey Folder

### Purpose
`.private-journey/` is a gitignored folder that acts as your private AI context layer. It keeps personal notes, decisions, and setup instructions out of the public repo while still being readable by Claude Code.

### Structure
```
.private-journey/
├── context.md      ← extra context Claude Code reads (current focus, preferences, decisions)
├── JOURNEY.md      ← dev log — what you tried, what failed, why you made each call
├── setup.md        ← personal env setup, API key notes, local shortcuts
└── decisions/      ← private ADRs (architecture decision records)
```

### How Claude Code reads it
Add this line at the bottom of `CLAUDE.md`:
```
## Private context
If `.private-journey/context.md` exists, read it for additional project context before responding.
```
Claude Code reads it when present, ignores it silently when absent — so public contributors are unaffected.

### What goes where

| Content | File | Committed? |
|---|---|---|
| Project rules, structure, commands | `CLAUDE.md` | ✅ Yes — public |
| Current focus, personal preferences | `.private-journey/context.md` | ❌ No — gitignored |
| Dev log, decisions journal | `.private-journey/JOURNEY.md` | ❌ No — gitignored |
| API keys, local setup notes | `.private-journey/setup.md` | ❌ No — gitignored |
| Personal Claude Code settings | `.claude/settings.local.json` | ❌ No — gitignored |

### .gitignore entries
```gitignore
# Private AI context
.private-journey/

# Personal Claude Code settings
.claude/settings.local.json

# Lottie runtime
.lottie/
```

---

## 8. Phase 0 Deliverables Checklist

- [ ] `lottie` CLI installable via `pip install lottie-orchestrator`
- [ ] `lottie init` creates a valid project with hello-world agent
- [ ] `lottie create agent` and `lottie create skill` scaffold all required files
- [ ] `BaseAgent` and `BaseSkill` abstract classes with auto-instrumentation
- [ ] `LLMProvider` abstraction wrapping litellm
- [ ] `lottie run` executes a single agent end-to-end
- [ ] `lottie benchmark` records all metrics to `.lottie/benchmarks/`
- [ ] `lottie doctor` validates environment
- [ ] CI/CD: mypy, ruff, pytest on every push
- [ ] `ScaffolderAgent` — the AI-powered generator agent
- [ ] `TemplateRendererSkill`, `SchemaValidatorSkill` built-in skills
- [ ] `LOTTIE.md` auto-updated by `lottie status`
- [ ] `CLAUDE.md` committed to repo root (public, clean)
- [ ] `.private-journey/` created locally and added to `.gitignore`
- [ ] `.private-journey/context.md` — personal context for Claude Code
- [ ] `.private-journey/JOURNEY.md` — dev log initialized

---

---

## 9. Testing Strategy

### The 4 Layers

| Layer | What | LLM? | Speed | Runs on | Command |
|---|---|---|---|---|---|
| 1 — Unit | Skills only | ✗ None | ⚡ <1s | Every save | `pytest skills/` |
| 2 — Integration | Agents (MockLLM) | 🟡 Mocked | 🏃 ~5s | Every PR | `pytest agents/` |
| 3 — Contract | Pydantic schemas | ✗ None | ⚡ <1s | Every PR | `pytest tests/contracts/` |
| 4 — Eval | Agent quality | ✓ Real LLM | 🐢 ~30s | Pre-release | `lottie benchmark agent` |
| 5 — E2E | Full pipelines | ✓ Real LLM | 🐢 ~2min | On tag only | `pytest tests/e2e/` |

### Layer 1 — Unit Tests (Skills)
Skills are deterministic — no LLM, no network. Test directly with real inputs.
Every skill generated by `lottie create skill` includes 3 baseline cases: happy path, edge case, error.
Located at `skills/<name>/tests/test_<name>.py`.

### Layer 2 — Integration Tests (Agents with MockLLMProvider)
Inject `MockLLMProvider` into agents. Returns pre-defined responses.
Tests the agent's **decision logic and tool calls**, not LLM quality.
Every agent must have 3 baseline cases: normal flow, tool call flow, policy violation flow.

```python
# MockLLMProvider usage
from lottie.llm import MockLLMProvider

def test_researcher_calls_search_tool():
    mock_llm = MockLLMProvider(
        responses=["I need to search for this. Calling WebSearchSkill..."]
    )
    agent = ResearcherAgent(llm=mock_llm)
    result = agent.run(ResearchInput(query="latest AI news"))
    assert mock_llm.tool_calls_made == ["WebSearchSkill"]
```

### Layer 3 — Eval Tests (Quality benchmarks)
Compare real LLM output against **golden outputs** in `evals/fixtures/`.
Run with `lottie benchmark agent <name>`. Results tied to git commit hash.
A drop in `accuracy_pct` below threshold blocks the GitHub Release via CI.

### Layer 4 — E2E Tests (Phase 2+)
Full pipeline smoke tests. Run only on release tags. 3 scenarios per phase milestone.

### Testing tools
- `pytest` + `pytest-asyncio` — test runner
- `pytest-cov` — coverage (≥80% required)
- `respx` — HTTP mocking for skills that call external APIs
- `MockLLMProvider` — built in Phase 0, used in all agent integration tests

---

## 10. GitHub Release Workflow

### Branching Model
| Branch | Purpose | Protection |
|---|---|---|
| `main` | Always releasable. Only merged from dev via PR. | 🔒 Protected |
| `dev` | Integration branch. All features merge here first. | PR required |
| `feat/*` | e.g. `feat/phase0-base-classes` | — |
| `fix/*` | e.g. `fix/llm-provider-timeout` | — |
| `chore/*` | e.g. `chore/update-deps` | — |

### Versioning — Semantic + Conventional Commits
| Tag | Phase | What ships |
|---|---|---|
| `v0.1.0` | Phase 0 | BaseAgent, BaseSkill, CLI, generators, MockLLMProvider |
| `v0.2.0` | Phase 1 | Knowledge Core — ChromaDB, RAG pipeline, policy store |
| `v0.3.0` | Phase 2 | Agent Mesh — LangGraph engine, supervisor, parallel runner |
| `v0.4.0` | Phase 3 | Governance — audit trail, policy engine, OpenTelemetry |
| `v0.5.0` | Phase 4 | Integration Layer — MCP server, OpenAI-compat API |
| `v1.0.0` | Phase 5 | Public SDK, docs site, plugin system, demo projects |

### Conventional Commit Format
```
feat(agents): add BaseAgent with auto-instrumentation    → minor bump
fix(llm): handle provider timeout gracefully             → patch bump
feat!: rename LLMProvider interface                      → major bump
docs(agent): update ResearcherAgent AGENT.md             → no bump
chore: update dependencies                               → no bump
```

### GitHub Actions — CI/CD Pipeline

**On every PR → dev:**
1. `ruff check` — lint
2. `mypy --strict` — typecheck
3. `pytest skills/` — unit tests
4. `pytest agents/` — integration tests
5. Coverage check ≥ 80%

**On tag `v*` (release):**
1. All above must pass
2. `lottie benchmark` — eval tests (accuracy gate)
3. `python -m build` — build wheel + sdist
4. Publish to PyPI via trusted publisher (no API keys in CI)
5. Create GitHub Release with auto-generated changelog

### Branch Protection Rules (main)
1. Require pull request — no direct pushes to main, ever
2. Require passing CI — all checks must pass before merge
3. Require linear history — squash-merge only, one clean commit per PR
4. Auto-delete branches after merge

---

## 11. Claude Code Setup — Git Author

### Step 1: Set your git identity globally
```bash
git config --global user.name "cdiaz"
git config --global user.email "cristian.diaz.jim@gmail.com"

# Verify
git config --global --list | grep user
```

### Step 2: Remove Claude Code co-author trailer
Create `~/.claude/settings.json` to apply globally to all projects:
```json
{
  "attribution": {
    "commit": "",
    "pr": ""
  }
}
```

Also commit `.claude/settings.json` to the Lottie repo so all contributors get clean commits:
```bash
mkdir -p .claude
cat > .claude/settings.json << EOF
{
  "attribution": {
    "commit": "",
    "pr": ""
  }
}
EOF
```

Add `.claude/settings.local.json` to `.gitignore` for personal Claude Code preferences.

### Result
```
# Before
feat(agents): add BaseAgent

Co-authored-by: Claude <noreply@anthropic.com>   ← removed

# After — just you
feat(agents): add BaseAgent
```

---

---

## 12. Security Layer

### Core principle
Security is a **cross-cutting layer**, not a phase. It sits at two points: the **input gate** (before anything reaches the orchestrator) and the **output gate** (before anything leaves Lottie). Every request is sanitised on the way in and validated on the way out. Agents never touch raw external input.

### OWASP LLM Top 10 — Lottie mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| LLM01 Prompt Injection | 🔴 Critical | `PromptInjectionScanSkill` on all inputs + all retrieved knowledge |
| LLM02 Insecure Output Handling | 🔴 Critical | `OutputValidationSkill` + `CodeSecurityScanSkill` (bandit) before any file write |
| LLM06 Sensitive Data Disclosure | 🔴 Critical | `SecretDetectionSkill` (detect-secrets) on all outputs and on knowledge ingest |
| LLM08 Excessive Agency | 🟠 High | Agents declare capabilities in `config.yaml`; runtime blocks undeclared skill calls |
| LLM07 Insecure Skill Design | 🟠 High | Pydantic v2 schemas on all skill inputs; dangerous skills require explicit capability declaration |
| LLM09 Overreliance | 🟠 High | Code review pipeline + human gate; knowledge write-protection (draft-only for agents) |
| LLM03 Knowledge Poisoning | 🟠 High | Ingest-time scan: injection + secret detection before any file enters knowledge layer |
| LLM04 Model DoS | 🟡 Medium | Cost budget limits per agent; circuit breaker; max_tokens_per_run in config.yaml |

### Five security checkpoints
1. **Ingest** — scan for injection patterns and secrets before storing in knowledge layer
2. **Runtime Input** — sanitize + policy check + capability enforce before agent receives input
3. **Runtime Output** — secret detection + code scan + schema validation before anything leaves Lottie
4. **Human Gate** — draft→curated always requires human review; high-risk operations require approval
5. **Audit** — everything logged; post-incident replay; every security event tied to agent + provider + timestamp

### Security module: `src/lottie/security/`
Built in **Phase 1** (not Phase 2 — security cannot wait):
```
prompt_injection.py    # regex + heuristic scanner, rejects on match
secret_detector.py     # detect-secrets library + custom patterns
output_validator.py    # Pydantic schema check + content scan
capability_enforcer.py # runtime: agent can only call declared skills
code_scanner.py        # bandit (Python), semgrep rules, mypy
input_sanitizer.py     # strips injection patterns, normalises encoding
```

### Security Skills (built in Phase 1)
| Skill | When applied |
|---|---|
| `PromptInjectionScanSkill` | All external inputs + all retrieved knowledge |
| `SecretDetectionSkill` | Knowledge ingest + all LLM outputs + before any file write |
| `OutputValidationSkill` | All LLM outputs — schema + content validation |
| `InputSanitizerSkill` | First step for all external inputs |
| `CodeSecurityScanSkill` | Generated/modified code — bandit + semgrep (Phase 2) |
| `CapabilityEnforcerSkill` | Before every skill invocation — checks agent config.yaml (Phase 2) |

### Agent capability declaration (prevents LLM08)
```yaml
# agents/researcher/config.yaml
provider: anthropic/claude-sonnet-4-6
capabilities:
  - WebSearchSkill        # can call this
  - DocumentIngestSkill   # can call this
  # anything NOT listed is blocked at runtime
file_access: read-only
network_access: allowed   # via declared skills only
max_tokens_per_run: 4000  # DoS protection
policies: [base, no-pii, no-secrets]
```

### Human gate triggers
- File writes outside project root
- Shell execution (any subprocess/os.system call)
- First network call to a new domain
- Knowledge promotion (draft → curated, always)
- Any policy rule violation (escalated, never silently dropped)

---

## 13. Multi-Agent Orchestration (LangGraph)

### Yes — LangGraph is confirmed as the orchestration engine.

LangGraph provides exactly what Lottie needs:
- **State checkpointing** — pause, resume, replay any workflow; essential for human-in-the-loop gates
- **Conditional routing** — explicit graph edges: policy violation → escalate, worker error → retry with different provider
- **Mixed LLM graphs** — each node declares its own provider; Researcher on Claude, Worker on GPT-4o, in the same graph
- **Audit-ready** — state transitions map 1:1 to Lottie's audit log; time-travel debugging replays any run

### Core pattern: Supervisor → Specialists
```
Human (interrupt/approve)
       ↕
  Supervisor Agent  (routes by intent, any LLM)
  ↙        ↓        ↘
Researcher  Worker  Reviewer   (each can use different LLM)
       ↓ (every call passes through)
Security Gate + Knowledge Layer
```

### LangGraph patterns used
| Pattern | Use |
|---|---|
| Supervisor-Worker | Hierarchical routing — Supervisor decides, Workers execute |
| Human-in-the-loop | `interrupt()` before high-risk actions or policy violations |
| Parallel branches | Fork/join for concurrent agent execution, typed result merge |
| Conditional edges | Route based on output content, policy result, or error type |
| Time-travel | Replay any past run for debugging or audit investigation |

---

## 14. Code Review Pipeline

Every time Lottie generates or modifies code, this pipeline runs in order:

1. **LLM generates** — via `LLMProvider` abstraction
2. **Secret scan** — `SecretDetectionSkill` (detect-secrets)
3. **Security scan** — `CodeSecurityScanSkill` (bandit + semgrep)
4. **Type check** — `mypy --strict`
5. **Lint + format** — `ruff check` + `ruff format`
6. **Tests** — `pytest` with `MockLLMProvider` (no real LLM in CI)
7. **Human gate** — for high-risk operations (shell exec, out-of-scope writes)

Steps 1–6 are automatic. Step 7 is triggered only when the capability enforcer flags a high-risk operation.

---

## 15. Updated Knowledge Architecture (Graph-first)

### Principle: files as persistence, graph as query layer

| Layer | Storage | Retrieval method |
|---|---|---|
| Layer 0 — global | YAML files | Always injected, deterministic |
| Layer 1 — platform | YAML files | Deterministic by project config |
| Layer 2 — project | YAML files | Deterministic by project config |
| Layer 3 — memory | YAML files → ChromaDB when >200 files | Tag-match (`yq` filter) or semantic |
| Layer 4 — ephemeral | In-memory only | Never persisted |

### YAML frontmatter on every knowledge file
```yaml
---
id: lottie/auth-conventions
layer: platform
scope: lottie
topic: authentication
tags: [auth, jwt, sessions]
status: curated         # draft | curated | aging | deprecated | archived
last_verified: 2026-05
depends_on: [global/conventions]
supersedes: []
---
```

### In-memory graph (~30 lines, networkx)
```python
import networkx as nx

def build_knowledge_graph(manifest: Manifest) -> nx.DiGraph:
    G = nx.DiGraph()
    for entry in manifest.entries:
        G.add_node(entry.id, **entry.metadata)
        for dep in entry.depends_on:
            G.add_edge(dep, entry.id, rel="depends_on")
    return G  # cycle detection, impact analysis, traversal — all free
```

### New `lottie memory` commands
```bash
lottie memory graph              # visualize dependency graph
lottie memory impact <file>      # what breaks if this is deprecated?
lottie memory audit              # find cycles, orphans, stale deps
lottie memory review             # surface drafts for human review
lottie memory add <file>         # scaffold with correct frontmatter
lottie knowledge ingest --format graphify ./graph.json  # import external graph
```

### GraphIngestSkill
Converts external graph tool output (Graphify, CodeGraph, JSON-LD, GraphML) into Lottie knowledge files with correct frontmatter. Runs `PromptInjectionScanSkill` and `SecretDetectionSkill` on every generated file before storing.

---

*Generated by Lottie AI Orchestrator · Phase 0 Spec · 2026*
