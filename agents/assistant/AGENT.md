# AssistantMesh

## Role
Reference **mesh** agent: a `MeshAgent` subclass (from `lottie.mesh`) that routes a single task between three existing workers — `research` (knowledge-grounded), `critic` (reviewer) and `publisher` (release finalizer). The injected LLM acts as the **supervisor**, picking the next worker at each step; the engine loops until the supervisor returns `FINISH` or `max_steps` is reached.

The `publisher` worker can optionally be wired as a **human-in-the-loop (HITL)** worker via `interrupt_before` (see below): when enabled, routing to it pauses the run for human approval before it executes. By default `interrupt_before` is **not** set, so the assistant runs on the langgraph-free `LocalEngine` and stays installable/runnable on a base install.

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
  - publisher
```

The `workers:` list in `config.yaml` is the **capability allow-set**: it is exactly the routing roster the supervisor may pick from. It must match the keys of `_DESCRIPTIONS` in `agent.py`. The supervisor router (`SupervisorRouter`) only ever returns one of these worker names or `FINISH` — anything else is a capability violation.

| Worker | Description |
|---|---|
| `research` | Retrieves and synthesizes knowledge to answer the task. |
| `critic` | Reviews the latest draft and suggests one concrete improvement. |
| `publisher` | Publishes/finalizes the answer for release (HITL — see below). |

## Human-in-the-Loop (`interrupt_before`) — opt-in

HITL is **opt-in** and is **not** enabled in the default `config.yaml`. To enable it, add an `interrupt_before:` list to `config.yaml` (or pass it via `AgentConfig`) naming the workers that should **pause for human approval** before they run, e.g.:

```yaml
interrupt_before:
  - publisher
```

When `interrupt_before` is non-empty, `from_project` builds the mesh on the `LangGraphEngine` (which checkpoints state) instead of the default `LocalEngine`. This requires the **`[mesh]` extra** (langgraph) to be installed; if it is not, `from_project` raises a clear `MeshError`. `interrupt_before` must be a subset of the mesh's worker adapters (`_DESCRIPTIONS`), enforced by the same consistency guard as `workers`.

By default — with no `interrupt_before` set — the assistant runs on `LocalEngine` and needs neither langgraph nor the `[mesh]` extra.

When the supervisor routes to `publisher`, the engine pauses **before** the node runs and `mesh.run(...)` returns `MeshOutput` with `status="interrupted"` and a populated `pending` (`PendingApproval(worker="publisher", ...)`). No publish LLM call has happened yet. The caller then resumes:

```python
out = mesh.run(AssistantInput(task="write an overview"))
if out.status == "interrupted":
    resumed = mesh.resume(out.thread_id, ApprovalDecision(action="approve"))
    # action="approve" runs the publisher node and continues the loop;
    # action="reject" records the rejection without executing the worker.
```

Once `interrupt_before: [publisher]` is enabled, the assistant therefore **pauses for approval** whenever the supervisor decides to publish. With the default `config.yaml` (no `interrupt_before`), the publisher runs without pausing.

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
