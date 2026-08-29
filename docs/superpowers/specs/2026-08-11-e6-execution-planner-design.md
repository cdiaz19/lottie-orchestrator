# E6 — Execution Planner (re-scoped)

> Epic design. Target: **v3.3.0**. Date: 2026-08-11.
> Theme: make a mesh run **reproducible** — record what it decided, render it, replay it
> without the supervisor.
> Follows `2026-07-30-v3-runtime-kernel-design.md` §8 (E6), **and re-scopes it**.

---

## 1. Architecture review — the premise did not survive

The V3 spec described E6 as:

> *"A typed `Plan` DAG that both single-agent (a 1-node plan) and mesh compile to.
> Unlocks `lottie plan <agent>` dry-run/explain, pre-execution cost estimation,
> deterministic replay."*

**The mesh has no plan. It has a routing loop.** `LocalEngine.run` calls `route(state)` on
**every step**, and `SupervisorRouter.route` builds its prompt from `state.history` — the
work already done. Decision N+1 depends on result N.

So a mesh **cannot be compiled to a DAG ahead of time**, and that is not a gap to close:
the dynamism *is* the feature. If the flow were known in advance you would not be paying an
LLM to route it.

That invalidates two of the three promised unlocks:

| Promised unlock | Reality against the code |
|---|---|
| `lottie plan` dry-run | A single agent is one node — trivially uninteresting. A mesh cannot be known without running it. |
| Pre-execution cost estimate | Requires knowing the path, which requires running the supervisor. |
| **Deterministic replay** | **Real and valuable** — but it needs a *recorded* plan, not a predicted one. |

The V3 spec predicted exactly this: *"the most likely to be re-scoped or split after its
own review."* This is that re-scope, recorded rather than quietly narrowed.

### What already exists

`MeshState.history` is `list[StepResult]` with `worker`, `result`, `step`. That is most of
an execution record already — what is missing is the *routing decisions* (notably which
steps were a parallel fan-out) and a stable identity for the run.

---

## 2. Decision

**A `Plan` is an execution artifact, not a prediction.** A run records the routing
decisions it made; `lottie plan show` renders them; **replay re-executes that recorded
sequence without calling the supervisor at all.**

| Rejected alternative | Why |
|---|---|
| Static declared plans as an alternative execution mode | A genuinely useful feature — it removes an LLM call per step — but it is a *new capability*, not the unification the spec described, and it leaves the dynamic case untouched. Better as its own epic if it is asked for. |
| Skip E6 entirely | Replay alone justifies the work: multi-agent flows are currently non-deterministic, which makes them untestable and expensive to debug. |

### What replay buys

- **Regression tests over multi-agent flows.** Today a mesh test either mocks the
  supervisor by hand or is non-deterministic. A recorded plan makes a real run repeatable.
- **Debugging without paying for routing.** Re-run a failure with zero supervisor calls.
- **A truthful "what happened" artifact** — the same discipline as the audit ledger, at
  the flow level.

---

## 3. Architecture

### 3.1 `PlanStep` / `Plan` (`mesh/plan.py`)

```python
class PlanStep(BaseModel):
    step: int
    workers: list[str]      # >1 == a parallel fan-out
    finished: bool = False  # the supervisor said FINISH here

class Plan(BaseModel):
    task_sha256: str        # binds a plan to the task it was recorded for
    steps: list[PlanStep]
    created_at: float
```

`task_sha256` rather than the task text: a plan is stored on disk and the same hash-only
discipline that governs the audit ledger applies — a recorded plan should not become a new
place raw task text accumulates.

### 3.2 `record_plan(result) -> Plan`

Derives a `Plan` from a finished `MeshRunResult`. Parallel fan-out is recoverable because
`StepResult.step` is a monotonic index that repeats across workers in the same fan-out —
grouping by `step` recovers the shape.

### 3.3 `replay_route(plan) -> RouteFn`

A `RouteFn` that yields the recorded decisions in order instead of asking the LLM. It
drops into the existing engine unchanged — `MeshEngine.run` takes `route` as a parameter,
which is exactly the seam replay needs, and neither engine changes.

**Fail-closed on divergence:** if replay is asked for a worker the mesh no longer declares,
it raises rather than silently skipping. A replay that quietly diverges from its recording
is worse than one that refuses.

### 3.4 CLI

- `lottie plan show <agent> --thread <id>` — render a recorded plan
- `lottie plan replay <agent> --thread <id>` — re-run it without the supervisor

---

## 4. Slice plan

| Slice | Delivers | Lab |
|---|---|---|
| **S1** | `Plan`, `record_plan`, `replay_route`, plan storage, `lottie plan show/replay` | **R37** |
| **S2** | Release: bump 3.3.0, CHANGELOG, tag | full regression |

---

## 5. Invariants

- **No engine changes.** Replay is a `RouteFn`; both engines take one already.
- **Replay makes zero supervisor calls.** Asserted directly — it is the whole point.
- **A plan stores a task hash, never the task text.**
- **Divergence fails closed**, never silently.
- Rule 7b gate, one PR, one lab round.

---

## 6. Definition of Done (v3.3.0)

- A mesh run can be recorded, rendered, and replayed with zero supervisor calls.
- Replay against a changed worker roster fails loudly.
- R37 green, full regression green, `v3.3.0` tagged.
