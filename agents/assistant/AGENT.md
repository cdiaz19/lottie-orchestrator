# AssistantMesh

## Role
Reference **mesh** agent: a `MeshAgent` subclass (from `lottie.mesh`) that routes a single task between two existing workers — `research` (knowledge-grounded) and `critic` (reviewer). The injected LLM acts as the **supervisor**, picking the next worker at each step; the engine loops until the supervisor returns `FINISH` or `max_steps` is reached.

## Input

| Field | Type | Default | Description |
|---|---|---|---|
| `task` | `str` | — | The task to route across workers |
| `max_steps` | `int` | `8` | Hard cap on routing steps before the loop ends |

`AssistantInput` is a discovery-named alias over `MeshInput`.

## Output

| Field | Type | Description |
|---|---|---|
| `final` | `str` | The final answer assembled by the mesh run |
| `history` | `list[StepResult]` | Ordered record of each worker invocation (`worker`, `result`) |

`AssistantOutput` is a discovery-named alias over `MeshOutput`.

## Workers (capability allow-set)

```yaml
workers:
  - research
  - critic
```

The `workers:` list in `config.yaml` is the **capability allow-set**: it is exactly the routing roster the supervisor may pick from. It must match the keys of `_DESCRIPTIONS` in `agent.py`. The supervisor router (`SupervisorRouter`) only ever returns one of these worker names or `FINISH` — anything else is a capability violation.

| Worker | Description |
|---|---|
| `research` | Retrieves and synthesizes knowledge to answer the task. |
| `critic` | Reviews the latest draft and suggests one concrete improvement. |

## How Routing Works

1. `AssistantMesh.from_project` builds the two workers — `ResearchAgent.from_project(...)` and `CriticAgent(llm, ...)` — and wraps each in a `MeshNode` adapter closing over the mesh instance.
2. `MeshAgent._execute` seeds a `MeshState(task=...)` and loops: the supervisor (the injected LLM, via `self.complete`) routes to the next worker given the `_DESCRIPTIONS`, the worker node runs and appends a `StepResult` to history.
3. The loop ends when the supervisor returns `FINISH` or `max_steps` is reached, returning `MeshOutput(final, history)`.

## Token / Cost Accumulation

Each worker node calls `self._accumulate(worker.last_metrics)` after running its worker, folding that worker's tokens and cost into the active mesh run context. The supervisor's own routing calls go through `self.complete` and are accumulated automatically. As a result, **all worker token/cost rolls up into the mesh's `last_metrics`** after `run()` completes.

## Provider
Default: `anthropic/claude-sonnet-4-6` (the supervisor LLM). Workers receive the same injected LLM.

## Policies
- `base`

## CLI

Reuses the existing stack — **no new CLI**:

```bash
lottie run assistant            # route a task through the mesh
lottie serve                    # exposes assistant as an MCP tool
lottie benchmark agent assistant
```

## Example

```python
from agents.assistant.agent import AssistantMesh
from agents.assistant.schema import AssistantInput

mesh = AssistantMesh.from_project(llm=llm, root=root, config=config)
out = mesh.run(AssistantInput(task="Summarize multi-agent AI systems.", max_steps=4))
print(out.final)
for step in out.history:
    print(step.worker, "->", step.result)
```
