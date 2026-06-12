# Phase 2 — Agent Mesh (v0.3.0) — Design

> Status: approved design, pre-plan. Source of truth for the Phase 2 implementation plan.
> Spec anchor: `LOTTIE_PHASE0_SPEC.md` §13 (Multi-Agent Orchestration). Roadmap row: `v0.3.0` — Agent Mesh.

## Goal

Ship the **core** multi-agent mesh: a supervisor agent that routes a task to declared worker
agents by intent, over conditional edges, with fully typed Pydantic state — testable end to end on
`MockLLMProvider`. A mesh is itself a `BaseAgent`, so it reuses the entire existing
run / serve / MCP / benchmark stack with zero changes to that stack.

## Scope

### Ships in v0.3.0
- Supervisor → worker routing (LLM intent, validated to the declared worker set).
- Conditional edges (route depends on supervisor decision; loop until `FINISH` or `max_steps`).
- Typed Pydantic `MeshState` flowing through every node (Rule 2 honored on the hot path).
- `MeshEngine` ABC + hand-rolled `LocalEngine` default (no heavy dep, MockLLM-testable).
- Capability enforcement: router rejects any target not in `config.yaml` `workers:`.
- Metrics rollup: mesh `RunResult` reports aggregated tokens / cost / latency across workers + supervisor.
- One reference mesh (`agents/assistant/`) with workers `[research, critic]`, integration-tested + benchmarkable.

### Deferred → Phase 3+ (explicitly out of scope)
- LangGraph backend (`LangGraphEngine` adapter behind the ABC).
- Parallel branches (fork/join, typed result merge).
- Human-in-the-loop `interrupt()` / resume (needs a persistent checkpointer).
- Time-travel replay.
- Per-node security gate (mesh boundary gate via `AgentService` covers in/out for now).
- Persistent checkpointer / state durability.

## Architecture

A mesh **is a `BaseAgent`**. It is discovered, instantiated (via the existing
`instantiate_agent` → `from_project` DI seam), run, served (one MCP tool per mesh), and benchmarked
exactly like any other agent. No changes to `AgentService`, `lottie run`, serve transports, or the
benchmark runner are required for a mesh to flow through them.

```
agents/<mesh>/agent.py        _execute(MeshInput) -> builds MeshState, then loops:
        |                       supervisor.complete -> RouteDecision{next in workers | FINISH}
        v
   MeshEngine (ABC)  ----  LocalEngine (hand-rolled, ~80 LOC, default in tests + prod v1)
        |                  \-- LangGraphEngine adapter  -> Phase 3
        v
   worker nodes = existing BaseAgents (research, critic, ...)
   typed adapters fold worker In/Out <-> MeshState
```

### New framework code — `src/lottie/mesh/`
- `state.py` — Pydantic models:
  - `MeshState{task:str, history:list[StepResult]=[], final:str|None=None}`
  - `StepResult{worker:str, result:str, metadata:dict[str,str]={}}`
  - `RouteDecision{next:str}` (`next` is a worker name or the `FINISH` sentinel)
- `engine.py` — `MeshEngine(ABC)`:
  `run(state:MeshState, nodes:Mapping[str, NodeFn], router:Router, *, max_steps:int) -> MeshState`
- `local.py` — `LocalEngine(MeshEngine)`: deterministic supervisor loop, conditional dispatch,
  `max_steps` guard (raises `MeshStepLimitExceeded` when exceeded without `FINISH`).
- `router.py` — `SupervisorRouter`: builds a routing prompt from worker names + descriptions, calls
  the injected `complete` callable, parses the response into `RouteDecision`, validates
  `next in declared_workers | {FINISH}`; an undeclared/hallucinated target raises `CapabilityViolation`.
- `base.py` — `MeshAgent` helper (subclass of `BaseAgent`): registers worker node adapters, wires the
  router + engine, and aggregates child run metrics into the active `RunContext`.
- `errors.py` (or in `engine.py`) — `MeshError` base, `CapabilityViolation`, `MeshStepLimitExceeded`.

`NodeFn = Callable[[MeshState], MeshState]` — a typed adapter that builds a worker's `Input` from
state, runs the worker, and folds its `Output` back into a new `MeshState`.

## Data flow (one run)

```
AgentService.run_agent(mesh, payload)        # gate-in, validate -> MeshInput
  -> MeshAgent._execute:
       state = MeshState(task=payload.task)
       step = 0
       while step < max_steps:
         decision = router.route(state, workers)     # LLM via self.complete
         if decision.next == FINISH: break
         node = nodes[decision.next]                  # validated in config.workers
         state = node(state)                          # typed adapter: build In -> run -> fold Out
         step += 1
       else:
         raise MeshStepLimitExceeded
       return MeshOutput(final=state.final, history=state.history)
  -> gate-out, RunResult                              # aggregated tokens / cost / latency
```

The mesh's `Input`/`Output` are Pydantic models (`MeshInput{task:str, max_steps:int=...}`,
`MeshOutput{final:str, history:list[StepResult]}`), so the existing gate-in / validate / gate-out
path in `AgentService.run_agent` applies without modification.

## Capability enforcement (a Phase 2 deliverable)

The mesh `config.yaml` `workers:` list is the capability allow-set. `SupervisorRouter` validates the
LLM's chosen `next` against it; anything outside the set raises `CapabilityViolation`. This delivers
the runtime capability-enforcement posture promised by spec §13 / CLAUDE.md rule 11 — scoped to mesh
routing in v0.3.0 (not yet every individual skill call, which remains a later item).

## Metrics rollup (framework touch)

Each `worker.run()` is its own instrumented run with its own `last_metrics`; the supervisor's
`self.complete` calls already accumulate into the mesh's `RunContext`. `MeshAgent._execute` folds each
worker's `last_metrics` (input/output tokens, cost, latency) into the active mesh `RunContext` so that
`AgentService` reports **true mesh totals**, not just supervisor cost. This is a small, contained touch
to the `core/` run-context rollup and resolves the Phase-1 FU-2-class gap for the mesh path.

## Reference mesh + testability

- Reference unit `agents/assistant/` — a `MeshAgent` with `config.yaml` `workers: [research, critic]`.
  - `research` already exists (Phase 1, knowledge-grounded).
  - `critic` — a **new minimal `BaseAgent`** that reviews / refines the current digest in `MeshState`,
    giving routing two real, distinct targets so conditional edges are demonstrable.
- Integration test (`MockLLMProvider`, no real LLM — Rule 5): scripted to route
  `research -> critic -> FINISH`. Asserts: typed `MeshOutput`, `history` order, aggregated metrics
  populated, `max_steps` guard fires when never reaching `FINISH`, `CapabilityViolation` on a bad route.
- `agents/assistant/evals.yaml` for `lottie benchmark agent assistant`.

## CLI / serve / docs

No new CLI surface is required. The following already work through existing wiring once the mesh is a
discoverable agent:
- `lottie run assistant --input '{"task": "..."}'`
- `lottie serve` — exposes an `assistant()` MCP tool alongside the per-agent tools.
- `lottie benchmark agent assistant`.

Optional polish: `lottie inspect agent assistant` surfaces the worker topology (read from `config.yaml`).

Close-out: docs/spec sync (mark `v0.3.0` Agent Mesh delivered in the release table), `CLAUDE.md`
project-structure note for `src/lottie/mesh/`, and cut the `v0.3.0` tag on merge to `main`.

## Decisions locked (from brainstorming)

| # | Decision | Choice |
|---|---|---|
| D1 | v0.3.0 scope | Core mesh only (supervisor + conditional edges + typed state + MockLLM-testable). Parallel/HITL/time-travel deferred. |
| D2 | Unit model | A mesh **is a `BaseAgent`** — reuses run/serve/MCP/benchmark unchanged. |
| D3 | Orchestration engine | `MeshEngine` ABC + hand-rolled `LocalEngine` default. `LangGraphEngine` adapter deferred to Phase 3. |
| D4 | Supervisor routing | LLM intent via `self.complete`, validated against declared `workers:` (= capability enforcement), bounded by `max_steps`. |
| D5 | State <-> worker | Typed per-worker adapter methods; `MeshState` is Pydantic; existing agents reused as-is, unchanged. |
| D6 | Second worker | New minimal `critic` `BaseAgent` so the reference mesh has >=2 real routing targets. |
| D7 | Metrics | Mesh aggregates worker + supervisor metrics into its `RunContext` (small `core/` touch). |

## Risks / unknowns

- **Routing prompt reliability.** The LLM must return a parseable `RouteDecision`. Mitigation:
  structured/constrained output + strict parse + `CapabilityViolation` on invalid; `MockLLMProvider`
  drives deterministic tests.
- **Infinite-loop / cost runaway.** A mis-routing supervisor could loop. Mitigation: `max_steps` guard
  (hard cap) + `MeshStepLimitExceeded`.
- **Metrics rollup touches `core/`.** Risk of regressing single-agent metrics. Mitigation: additive
  rollup behind the mesh path; extend existing `core` metrics tests to prove single-agent runs unchanged.
- **Engine ABC churn when LangGraph lands.** The ABC must be shaped so a `LangGraphEngine` fits later
  without reworking `LocalEngine`'s callers. Mitigation: keep `MeshEngine.run` signature
  engine-agnostic (state in, state out; nodes + router injected).
- **Reference `critic` scope creep.** Keep `critic` minimal — a single-LLM-call refine/review agent,
  no knowledge layer — so it demonstrates routing without becoming its own project.

## Phase 2 — known follow-ups (deferred during this build)

| # | Item | Phase |
|---|---|---|
| FU-1 | `LangGraphEngine` adapter + persistent checkpointer | Phase 3 |
| FU-2 | Parallel branches (fork/join, typed result merge) | Phase 3 |
| FU-3 | Human-in-the-loop `interrupt()` / resume | Phase 3 |
| FU-4 | Time-travel replay (depends on checkpointer) | Phase 3 |
| FU-5 | Per-node security gate (every node call re-gated) | Phase 3 |
| FU-6 | Capability enforcement extended to every skill call (not just mesh routing) | Phase 3 |
