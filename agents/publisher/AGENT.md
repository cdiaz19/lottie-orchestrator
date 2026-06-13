# PublisherAgent

## Role
A terse **publisher** worker for the `assistant` mesh. Given a finished draft, it publishes/finalizes the text — returning the clean, release-ready final answer. It is knowledge-free and stateless: no knowledge layer, no skills, no `from_project` factory.

In the `assistant` mesh, `publisher` is a routable worker that **can optionally** be wired as an **`interrupt_before` HITL worker**: when enabled, routing to it pauses the run for human approval before the publish LLM call executes. HITL is opt-in (requires the `[mesh]` extra / `LangGraphEngine`); by default the assistant runs on `LocalEngine` and `publisher` runs without a pause.

## Input

| Field | Type | Description |
|---|---|---|
| `text` | `str` | The reviewed draft to publish/finalize |

## Output

| Field | Type | Description |
|---|---|---|
| `published` | `str` | The release-ready final text |

## How It Works

`PublisherAgent._execute` makes **exactly one** LLM call via `self.complete([...])`:

1. A `system` message carrying the publisher instructions (`SYSTEM_PROMPT`).
2. A `user` message carrying the draft (`data.text`).

The response content is wrapped in `PublisherOutput(published=...)` and returned. All LLM access goes through `self.complete` (CLAUDE.md rule 1), so tokens and cost are auto-accumulated into `agent.last_metrics` after `run()` completes.

## Token Accumulation

The single `self.complete` call forwards to the injected `LLMProvider` and records usage on the active run context. After `run()`, `agent.last_metrics` is populated.

## Provider
Default: `anthropic/claude-sonnet-4-6`

## Skills Used (capabilities)
_None._ The agent makes no skill calls — `capabilities` is empty.

## Policies
- `base`

## Role in the Mesh

`PublisherAgent` is a worker node in the `assistant` mesh. If `publisher` is added to that agent's `interrupt_before` (opt-in, needs the `[mesh]` extra), the `LangGraphEngine` pauses **before** the node runs and the mesh returns `status="interrupted"` with a `pending` approval; a human `approve` resumes the checkpoint and runs the single publish LLM call, while a `reject` records the rejection without executing the call. Without `interrupt_before` (the default), `publisher` runs inline like any other worker. The agent itself is unaware of the mesh — it only publishes the text it is handed.

## Examples

### Example 1 — publish a draft
```python
agent = PublisherAgent(llm)
out = agent.run(PublisherInput(text="A reviewed draft about multi-agent systems."))
print(out.published)
```
