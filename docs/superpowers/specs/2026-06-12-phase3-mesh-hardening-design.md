# Phase 3 — Mesh Hardening (v0.4.0) — Design

> Status: approved design, pre-plan. Source of truth for the Phase 3 mesh-hardening implementation plan.
> Builds on: `docs/superpowers/specs/2026-06-12-phase2-agent-mesh-design.md` (Phase 2 core mesh, PR #8).
> Roadmap note: the original roadmap had Phase 3 = Governance (v0.4.0). This cycle instead hardens the
> mesh; governance shifts out. The `v0.4.0` tag is reused for this work.

## Goal

Harden the Phase-2 agent mesh by adopting **LangGraph** as a real orchestration backend behind the
existing `MeshEngine` ABC, and delivering the three capabilities Phase 2 deferred: **time-travel/replay**
(via a checkpointer), **parallel fork/join**, and **human-in-the-loop (HITL) interrupt/resume**. The
hand-rolled `LocalEngine` remains the zero-dependency default; LangGraph is opt-in.

## Scope

All three capabilities ship this cycle, in one spec, internally sequenced as sub-phases A → B → C so
each is independently shippable and reviewable. If sub-phase C (HITL) slips, A + B still deliver value.

### Ships in v0.4.0
- **Sub-phase A — LangGraphEngine foundation:** `LangGraphEngine(MeshEngine)` compiling a
  `StateGraph(MeshState)`; checkpointer (MemorySaver default, SqliteSaver for persistence); parity with
  `LocalEngine`'s supervisor loop; time-travel = list/replay checkpoints by `thread_id`.
- **Sub-phase B — parallel fork/join:** supervisor may fan out to multiple workers; concurrent dispatch
  via LangGraph; typed, deterministically-ordered history merge.
- **Sub-phase C — HITL interrupt/resume:** config-declared `interrupt_before` workers pause the run,
  checkpoint, and surface a pending approval; `MeshAgent.resume(...)` continues from the checkpoint.
- Engine ABC + `MeshAgent`/`MeshOutput` contract extended (back-compatible defaults).
- `langgraph` packaged as an optional `[mesh]` extra.

### Deferred (out of scope)
- Policy-driven interrupts (belongs to the Governance phase / policy engine).
- Supervisor-emitted dynamic INTERRUPT (we use static config `interrupt_before`).
- Parallel/HITL on `LocalEngine` (LangGraphEngine-only; LocalEngine raises a clear error).
- A web/REST resume UI (CLI + programmatic only).

## Decisions locked (from brainstorming)

| # | Decision | Choice |
|---|---|---|
| D1 | Substrate | Adopt LangGraph behind the `MeshEngine` ABC; `LocalEngine` stays the zero-dep default. |
| D2 | This cycle's scope | All three (foundation + parallel + HITL), one spec, sub-phases A→B→C. |
| D3 | HITL trigger | Config-declared `interrupt_before: [worker,...]` → LangGraph native `interrupt_before`. |
| D4 | Packaging | Optional `[mesh]` extra; `LangGraphEngine` import-guarded; tests skipif-guarded; CI installs `.[mesh]`. |
| D5 | State bridge | Use `MeshState` directly as the LangGraph state schema; `history` becomes a reduced channel (`Annotated[list[StepResult], operator.add]`). No dict mirror. |
| D6 | Parallel fan-out | Additive `RouteDecision.parallel: list[str] = []` (single-route `next: str` unchanged); LangGraph `Send` fan-out; `StepResult.step: int` for deterministic merge order. |
| D7 | HITL contract | New `MeshRunResult{state, status, thread_id, pending}`; `MeshOutput` gains `status`/`thread_id`/`pending` (back-compatible defaults); `MeshAgent.resume()`. |
| D8 | Checkpointer | `MemorySaver` default; `SqliteSaver` at `.lottie/mesh/checkpoints.db` when persistence requested. |
| D9 | CLI | Minimal `lottie mesh resume <thread_id> --decision ... [--input ...]` and `lottie mesh history <thread_id>`. |

## Architecture

### Packaging & engine selection
- `pyproject.toml`: `[project.optional-dependencies] mesh = ["langgraph>=<pin>"]`.
- `src/lottie/mesh/langgraph_engine.py` import-guards `langgraph` at module top; if absent, raise a
  `MeshError` with the install hint (`pip install lottie-orchestrator[mesh]`) when the engine is
  constructed (not at import of `lottie.mesh`).
- `LocalEngine` remains the default in `MeshAgent.__init__` (`engine or LocalEngine()`).
- Parallel + HITL are LangGraphEngine-only. `LocalEngine.resume(...)` and any `interrupt_before`
  configuration with `LocalEngine` raise `MeshError("HITL/parallel requires LangGraphEngine; install .[mesh]")`.

### State bridge (Pydantic-native)
- `MeshState` is the LangGraph state schema directly (LangGraph accepts a Pydantic `BaseModel`).
- The only change to Phase-2 schema: `history: Annotated[list[StepResult], operator.add]` so concurrent
  branches append without clobbering. The annotation's default stays `[]` — Phase-2 sequential behavior
  and existing tests are unaffected.
- No TypedDict mirror, no per-node conversion. (Fallback if the pinned LangGraph version rejects Pydantic
  state + reducers: a thin TypedDict mirror converted at the engine boundary — see Risks.)

### Engine ABC + MeshAgent contract
New/changed types in `src/lottie/mesh/schema.py`:
```python
class PendingApproval(BaseModel):
    worker: str
    proposed_input: dict[str, str] = {}   # human-readable preview of the about-to-run worker input

class ApprovalDecision(BaseModel):
    action: Literal["approve", "reject"]
    edited_input: dict[str, str] = {}      # optional human edits applied on approve

class MeshRunResult(BaseModel):
    state: MeshState
    status: Literal["complete", "interrupted"] = "complete"
    thread_id: str | None = None
    pending: PendingApproval | None = None
```
`MeshOutput` gains (back-compatible defaults):
```python
class MeshOutput(BaseModel):
    final: str
    history: list[StepResult] = []
    status: Literal["complete", "interrupted"] = "complete"
    thread_id: str | None = None
    pending: PendingApproval | None = None
```
`MeshEngine` ABC (in `src/lottie/mesh/engine.py`):
```python
class MeshEngine(ABC):
    @abstractmethod
    def run(self, initial, *, nodes, route, max_steps, thread_id=None) -> MeshRunResult: ...
    @abstractmethod
    def resume(self, thread_id, *, nodes, route, decision) -> MeshRunResult: ...
```
- `LocalEngine.run` returns `MeshRunResult(state=<final>, status="complete")`; `LocalEngine.resume`
  raises `MeshError`. (Existing `LocalEngine` loop body is unchanged; only the return is wrapped.)
- `MeshAgent._execute` unwraps `MeshRunResult` → `MeshOutput` (carrying `status`/`thread_id`/`pending`).
- `MeshAgent.resume(thread_id: str, decision: ApprovalDecision) -> MeshOutput` delegates to
  `engine.resume(...)`. (Runs through the same `BaseAgent` instrumentation as a fresh run.)

### Sub-phase A — LangGraphEngine foundation + checkpointer + time-travel
- `LangGraphEngine(MeshEngine)` builds a `StateGraph(MeshState)`:
  - a **supervisor** node wrapping the injected `route` callable (returns the next worker / FINISH),
  - one node per entry in `nodes`,
  - conditional edges supervisor → worker / → END,
  - the `history` reducer accumulates `StepResult`s.
- Compiled with a checkpointer; `thread_id` (generated per run if not supplied) keys the run in the
  checkpointer config (`{"configurable": {"thread_id": ...}}`).
- Checkpointer selection: `MemorySaver` by default; `SqliteSaver` at `.lottie/mesh/checkpoints.db` when
  the engine is constructed with persistence enabled (constructor flag / factory).
- **Parity:** the same scripted routing through `LangGraphEngine` and `LocalEngine` produces identical
  `history` worker order and `final`.
- **Time-travel/replay:** list a thread's checkpoints and replay/inspect by `thread_id`, exposed
  programmatically and via `lottie mesh history <thread_id>`.

### Sub-phase B — parallel fork/join
- `RouteDecision.parallel: list[str] = []` (additive; `next: str` unchanged). When the supervisor returns
  a non-empty `parallel`, the engine fans out to those worker nodes concurrently (LangGraph `Send`).
- Each worker appends its `StepResult` via the `history` reducer; the join returns to the supervisor.
- `StepResult.step: int = 0` (dispatch index) makes merged history **deterministically sortable** despite
  completion-order nondeterminism. The supervisor router stamps the index when it emits a decision.
- Capability enforcement still applies: every name in `parallel` must be in the declared worker set
  (router validates, raising `CapabilityViolation` otherwise).

### Sub-phase C — HITL interrupt/resume
- `AgentConfig` gains `interrupt_before: list[str] = []` (additive). The names must be a subset of
  `workers` (validated in `from_project`, like the existing workers/descriptions guard).
- `LangGraphEngine` compiles with `interrupt_before=[<those node names>]`. When the supervisor routes to
  such a worker, LangGraph pauses **before** running it; the engine checkpoints and returns
  `MeshRunResult(status="interrupted", thread_id, pending=PendingApproval{worker, proposed_input})`.
- `MeshAgent.run` → returns a `MeshOutput(status="interrupted", thread_id, pending=...)` (no `final` yet).
- `MeshAgent.resume(thread_id, ApprovalDecision)`:
  - `approve` (optionally with `edited_input`) → continue from the checkpoint, run the worker, resume the loop.
  - `reject` → skip the worker, record a `StepResult(worker, result="rejected by human")`, resume routing.
- `AgentService.run_agent` surfaces `status`/`thread_id`/`pending` in `RunResult`; new
  `AgentService.resume_agent(name, thread_id, decision) -> RunResult`.
- CLI: `lottie run <mesh>` prints "awaiting approval for <worker>" + `thread_id` when interrupted;
  `lottie mesh resume <thread_id> --decision approve|reject [--input '<json>']` continues;
  `lottie mesh history <thread_id>` lists checkpoints.

## Components / file structure

**New:**
- `src/lottie/mesh/langgraph_engine.py` — `LangGraphEngine`, checkpointer wiring, StateGraph build, parallel `Send`, `interrupt_before` compile, resume.
- `src/lottie/cli/mesh.py` — `lottie mesh resume|history` Typer sub-app.
- `src/lottie/mesh/tests/test_langgraph_engine.py`, `.../test_parallel.py`, `.../test_hitl.py`.
- `tests/contracts/test_mesh_hardening_schema.py`.

**Modified:**
- `src/lottie/mesh/schema.py` — `MeshRunResult`, `PendingApproval`, `ApprovalDecision`; `MeshOutput` +fields; `StepResult.step`; `RouteDecision.parallel`; `history` reducer annotation.
- `src/lottie/mesh/engine.py` — ABC `run` returns `MeshRunResult`, add `resume`.
- `src/lottie/mesh/local.py` — wrap return in `MeshRunResult`; `resume`/interrupt → `MeshError`.
- `src/lottie/mesh/base.py` — `_execute` unwraps `MeshRunResult`; add `MeshAgent.resume`; thread_id plumbing.
- `src/lottie/mesh/router.py` — stamp `StepResult.step`; validate `parallel` names against the declared set.
- `src/lottie/mesh/__init__.py` — export new symbols.
- `src/lottie/project/config.py` — `AgentConfig.interrupt_before`.
- `src/lottie/serve/service.py` + `schema.py` — surface interrupted status; `resume_agent`.
- `src/lottie/cli/app.py` — register `mesh` sub-app.
- `agents/assistant/` — add a third `interrupt_before` worker to exercise HITL end-to-end; `config.yaml` gains `interrupt_before`.
- `pyproject.toml` — `[mesh]` extra; CI workflow installs `.[mesh]`.

## Testing

All hermetic (`MockLLMProvider`, `MemorySaver`, mock embeddings), skipif-guarded on `langgraph`.
- **A:** parity (LangGraphEngine vs LocalEngine identical history/final on the same route script);
  checkpoint persists + replays by `thread_id`.
- **B:** fan-out to ≥2 workers merges all `StepResult`s; deterministic order via `step`; `parallel` with an
  undeclared worker raises `CapabilityViolation`.
- **C:** route→`interrupt_before` worker → `status="interrupted"` + `pending`; `resume(approve)` completes
  with the worker's result; `resume(reject)` skips it; `LocalEngine` + HITL config → `MeshError`.
- Contract round-trips for `MeshRunResult`, `PendingApproval`, `ApprovalDecision`, extended `MeshOutput`/`RouteDecision`.
- Regression: Phase-2 mesh tests + serve/run tests stay green (back-compatible defaults).

## Risks / unknowns

- **LangGraph API/version drift.** Heavy dep, occasional breaking releases. Pin a version; isolate all
  LangGraph usage in `langgraph_engine.py` behind the ABC; `LocalEngine` is unaffected.
- **Pydantic-state + reducer support is version-dependent.** Verify in sub-phase A's first task that the
  pinned LangGraph accepts a Pydantic `BaseModel` state schema with an `Annotated[..., operator.add]`
  reducer. Fallback: a thin TypedDict mirror converted at the engine boundary (keeps `MeshState` the
  public type, dict only inside the adapter).
- **Contract-change blast radius.** `MeshOutput`/ABC change touches `MeshAgent`, `AgentService`, CLI.
  Mitigate with back-compatible defaults (`status="complete"`, `thread_id`/`pending`=`None`) and by
  extending existing serve/run tests to prove the sync path is unchanged.
- **Parallel determinism.** Completion order is nondeterministic; `StepResult.step` + sort gives a stable
  history. Tests assert set-membership + sorted order, never raw append order.
- **Resume statelessness in serve.** `resume_agent` must reconstruct the mesh (via `from_project`) and
  point the engine at the existing `thread_id`/checkpointer; with `MemorySaver` the checkpoint is
  process-local, so cross-process resume needs `SqliteSaver`. Document this; default CLI uses Sqlite for
  resume to survive process exit.

## Phase 3 — known follow-ups / deferred

| # | Item | Phase |
|---|---|---|
| FU-1 | Policy-driven interrupts (cost/rule predicates) | Governance phase |
| FU-2 | Supervisor-emitted dynamic INTERRUPT decisions | later |
| FU-3 | Parallel/HITL support in `LocalEngine` (no-dep) | later, if demanded |
| FU-4 | REST/web resume surface (beyond CLI) | Integration phase |
| FU-5 | Skill-internal LLM token accounting in `last_metrics` (carried from Phase 1 FU-2 / Phase 2) | later |
