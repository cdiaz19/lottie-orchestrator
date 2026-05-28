# LOTTIE — AI Orchestrator
Provider-agnostic multi-agent framework with shared knowledge & governance.

## Rules (read before writing any code)
- All LLM calls go through `lottie.llm.LLMProvider` — never import anthropic/openai directly
- All agent/skill inputs/outputs are Pydantic v2 models in schema.py
- Every agent needs AGENT.md, every skill needs SKILL.md
- Run `lottie create agent <name>` to scaffold — never create files manually
- All tests: `pytest`. Unit tests must not call real LLMs.

## Structure
- `src/lottie/` — core framework
- `agents/` — user-defined agents (each a self-contained module)
- `skills/` — user-defined skills (stateless, deterministic)
- `knowledge/` — raw docs for the shared knowledge base
- `policies/` — YAML governance rules
