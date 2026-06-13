# CriticAgent

## Role
A terse reviewer worker for the `assistant` mesh. Given a draft, it returns a short critique — what is accurate, what is missing, and one concrete improvement. It is knowledge-free and stateless: no knowledge layer, no skills, no `from_project` factory.

## Input

| Field | Type | Description |
|---|---|---|
| `text` | `str` | The draft to review |

## Output

| Field | Type | Description |
|---|---|---|
| `review` | `str` | A terse critique of the draft |

## How It Works

`CriticAgent._execute` makes **exactly one** LLM call via `self.complete([...])`:

1. A `system` message carrying the reviewer instructions (`SYSTEM_PROMPT`).
2. A `user` message carrying the raw draft (`data.text`).

The response content is wrapped in `CriticOutput(review=...)` and returned. All LLM access goes through `self.complete` (CLAUDE.md rule 1), so tokens and cost are auto-accumulated into `agent.last_metrics` after `run()` completes.

## Token Accumulation

The single `self.complete` call forwards to the injected `LLMProvider` and records usage on the active run context. After `run()`, `agent.last_metrics` is populated.

## Provider
Default: `anthropic/claude-sonnet-4-6`

## Skills Used (capabilities)
_None._ The agent makes no skill calls — `capabilities` is empty.

## Policies
- `base`

## Role in the Mesh

`CriticAgent` is a worker node in the `assistant` mesh (built later). The mesh supervisor routes a draft to the critic and receives a terse review back, which it can fold into further planning or revision steps. The agent itself is unaware of the mesh — it only reviews the text it is handed.

## Examples

### Example 1 — review a draft
```python
agent = CriticAgent(llm)
out = agent.run(CriticInput(text="A long draft about multi-agent systems."))
print(out.review)
```
