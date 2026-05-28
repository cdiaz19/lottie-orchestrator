# LOTTIE — AI Orchestrator
> Provider-agnostic multi-agent framework with shared knowledge & AI governance.
> Works with Claude Code, Cursor, Codex, and any LLM.

---

## Rules — read before writing any code

1. **Never import an LLM SDK directly.** All LLM calls go through `lottie.llm.LLMProvider`. Never use `anthropic`, `openai`, or any SDK directly in agent or skill code.
2. **All inputs/outputs are Pydantic v2 models.** Defined in `schema.py`. No raw dicts or strings crossing agent/skill boundaries.
3. **Every agent needs `AGENT.md`. Every skill needs `SKILL.md`.** Write the doc before the code.
4. **Use `lottie create agent <name>` or `lottie create skill <name>`** to scaffold. Never create agent/skill files manually.
5. **Unit tests must not call real LLMs.** Use `MockLLMProvider` from `lottie.llm`. Real LLM calls only in eval tests.
6. **Every file must pass `mypy --strict`.** No `Any` types without explicit justification.
7. **Commit convention:** `feat:`, `fix:`, `docs:`, `chore:`, `test:` — conventional commits only.

---

## Project structure

```
src/lottie/         — core framework (LLMProvider, BaseAgent, BaseSkill, CLI)
agents/             — user-defined agents (each a self-contained module)
skills/             — user-defined skills (stateless, deterministic)
knowledge/          — raw docs for the shared knowledge base
policies/           — YAML governance rules (allow/deny/escalate)
tests/              — contracts, e2e tests
.lottie/            — runtime (gitignored): chroma, audit.db, benchmarks
```

---

## Key abstractions

- `lottie.llm.LLMProvider` — abstract LLM interface (swap providers via config)
- `lottie.llm.MockLLMProvider` — for tests
- `lottie.core.BaseAgent` — all agents extend this
- `lottie.core.BaseSkill` — all skills extend this
- `lottie.core.BaseAgent` auto-instruments every run: latency, tokens, cost

---

## Running

```bash
lottie run <agent>                     # run an agent
lottie create agent <name>             # scaffold a new agent
lottie create skill <name>             # scaffold a new skill
lottie benchmark agent <name>          # benchmark an agent
lottie serve --port 8080               # start all integration endpoints
pytest                                 # run all tests
```

---

## Private context

If `.private-journey/context.md` exists, read it for additional project context before responding.
