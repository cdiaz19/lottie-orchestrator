# Phase 4 — Durable Resume over REST — Design

> Expose `POST /v1/agents/{name}/resume` on the shared HTTP app, backed by a **durable sqlite
> checkpointer** so a mesh interrupt can be resumed across server restarts and workers — not just
> within the one process that ran it. Closes the carried Phase-3 deferral (FU-9). Reuses the
> existing fail-closed SecurityGate + audit/policy/cost path — no second gate.

- **Date:** 2026-06-19
- **Phase:** Phase 4 (integration), slice 3 — after OpenAI-compat (PR #15) and generic REST (PR #16),
  both on `main`.
- **Branch:** `feat/durable-resume` (off `main`).

---

## 1. Goal & scope

The generic-REST slice surfaces a mesh interrupt on `POST /v1/agents/{name}/run` as
`status="interrupted"` + `thread_id` + `pending`, but deferred the resume endpoint because the
in-memory checkpointer is process-local: an in-memory resume only works under `uvicorn --workers 1`
until the next restart. Over HTTP that single-long-lived-process assumption is the **main** use case
failing, not a corner case — so durability is a prerequisite, not an enhancement. This slice bundles
both: durable cross-process resume **and** the resume transport.

**In scope:**
- A durable **sqlite** checkpointer for served mesh agents, selected by env, pointing at a shared
  root-derived db (`.lottie/mesh/checkpoints.db`).
- Resume that **rehydrates by `thread_id` from the checkpoint store**, not from the in-process agent
  cache — so a fresh worker / post-restart process can resume.
- `POST /v1/agents/{name}/resume` with a clean typed error contract.

**Out of scope (deferred):**
- **Streaming** (`BaseAgent` is sync-first; carried deferral).
- **`edited_input` application** on approve — accepted in the body but NOT applied to the resumed
  worker (the existing `LangGraphEngine.resume` simplification; documented, kept as-is).
- **Distributed (multi-host) resume** — requires a shared filesystem/db; not arbitrarily distributed.
- Auth / rate limiting.

**Locked decisions (resolved in brainstorming + confirmed against the code, do not relitigate):**
- **Checkpoint backend = env-driven default, set by serve.** Precedence: constructor arg >
  `LOTTIE_MESH_CHECKPOINT` env > `"memory"`. `lottie serve --port` sets `sqlite` once at startup. CLI
  /tests stay memory.
- **Durable resume = shared sqlite db + rehydrate-by-`thread_id`** (the agent cache stops being
  load-bearing).
- **Clean typed error map** (clients are programmatic HITL drivers): two new `ServeError` leaves
  `NotResumable` / `ThreadNotFound`, one new mesh leaf `ThreadNotFoundError`.
- **404 `thread_not_found`, not 410** — can't distinguish never-existed from pruned; 404 is the honest
  single code.
- **No second gate** — resume reuses `AgentService.resume_agent` → `BaseAgent.run` path.

## 2. Findings from grounding (de-risked the design)

A throwaway probe (a sqlite-backed `LangGraphEngine`, run→interrupt, then resume from a **fresh**
agent) established:
- **Durability works:** a fresh `MeshAgent` built with `checkpoint="sqlite"` + the same root found the
  interrupt's checkpoint by `thread_id` from disk — no shared in-memory state. Cross-process rehydrate
  is real.
- **Unknown-thread detection (FG-1):** a bogus `thread_id` surfaces *inconsistently* downstream —
  approve → `langgraph.errors.EmptyInputError`; reject → a pydantic `ValidationError` from our own
  `MeshState.model_validate({})` on an empty snapshot. Catching those two is fragile. **Clean signal:**
  `graph.get_state(config)` for a bogus thread returns an empty snapshot (no `task` in `values`).
  The engine pre-checks this and raises a typed error **before** attempting resume — action-independent.
- **Residual (FU-6):** sqlite deserialize of `StepResult` logs a langgraph msgpack-deprecation warning
  ("blocked in a future version … add to allowed_msgpack_modules"). A warning today; registering the
  type is a follow-up, noted not fixed.

## 3. Checkpoint selection — `LangGraphEngine` + `build_checkpointer`

`LangGraphEngine.__init__` currently defaults `checkpoint: str = "memory"`. Change the resolution so an
unspecified checkpoint reads the env:

```python
def __init__(self, *, checkpoint: str | None = None, root: Path | None = None, interrupt_before=None):
    ...
    self._checkpoint = _resolve_checkpoint(checkpoint)   # arg > env > "memory"
    self._root = root
```

```python
def _resolve_checkpoint(arg: str | None) -> str:
    """Precedence: explicit arg > LOTTIE_MESH_CHECKPOINT env > 'memory'."""
    if arg is not None:
        return arg
    return os.getenv("LOTTIE_MESH_CHECKPOINT", "memory")
```

`build_checkpointer("sqlite", root)` already exists (SqliteSaver under `<root>/.lottie/mesh/`); remove
its `# pragma: no cover` once a test exercises it. The db file path is **root-derived** so all workers /
a post-restart process share it. (Confirm the existing path is `.lottie/mesh/checkpoints.db`; keep it.)

Existing mesh agents that build `LangGraphEngine(interrupt_before=[...])` (no explicit checkpoint, e.g.
the lab's EditorMesh) automatically pick up the env → memory in CLI/tests, sqlite when served. **Zero
agent-author change.**

`cli/serve.py` `--port` branch sets the env once, before serving:

```python
    os.environ.setdefault("LOTTIE_MESH_CHECKPOINT", "sqlite")  # served meshes persist; set once
    uvicorn.run(build_http_app(root), host=host, port=port)
```

(`setdefault` so an operator can override to `memory`.) This mutates process-global env — acceptable for
a server process; called out in the docstring.

## 4. Unknown-thread detection — `ThreadNotFoundError` in the engine

`mesh/errors.py` gains `class ThreadNotFoundError(MeshError)`. `LangGraphEngine.resume` pre-checks the
snapshot before doing any work:

```python
    def resume(self, thread_id, *, nodes, route, decision) -> MeshRunResult:
        graph = self._build(nodes, route)
        config = {"configurable": {"thread_id": thread_id}}
        snap = graph.get_state(config)
        if not snap.values.get("task"):          # no real checkpoint for this thread
            raise ThreadNotFoundError(f"no checkpoint for thread {thread_id!r}")
        ... # existing reject/approve/invoke/_snapshot flow unchanged
```

This catches both downstream failure modes uniformly and never lets a raw langgraph/pydantic exception
leak (the FG-1 / yaml-ParserError discipline). The `LocalEngine.resume` (zero-dep path) is updated to
raise the same `ThreadNotFoundError` for an unknown thread, so the contract holds regardless of engine.

## 5. Service layer — `AgentService.resume_agent`

`serve/errors.py` gains two leaves:

```python
class NotResumable(ServeError):
    """The agent exists but cannot be resumed (not a mesh / no HITL)."""

class ThreadNotFound(ServeError):
    """No checkpoint exists for the given thread_id (never existed or pruned)."""
```

`resume_agent` maps mesh/agent failures to these, and gains the **same `_check_output` metrics-on-
withhold** treatment `run_agent` already has (so a withheld resume reports usage):

```python
    def resume_agent(self, name, thread_id, decision) -> RunResult:
        self._require_agent(name)                         # AgentNotFoundError -> 404
        agent = self._get_agent(name, None)               # rebuild by name (rehydrate seam)
        resume = getattr(agent, "resume", None)
        if resume is None:
            raise NotResumable(f"agent '{name}' is not resumable (not a mesh)")
        try:
            output = resume(thread_id, decision)
        except ThreadNotFoundError as exc:                # mesh leaf -> serve leaf (typed)
            raise ThreadNotFound(f"thread '{thread_id}' not found") from exc
        except Exception as exc:  # noqa: BLE001
            raise AgentExecutionError(f"agent '{name}' failed: {exc}") from exc
        self._check_output(agent, output)                 # output withhold -> OutputSecurityViolation(+metrics)
        return self._result(name, output, agent.last_metrics)
```

`_require_agent`/`_get_agent` already rebuild the agent by name; with the sqlite saver wired by env, a
**fresh `AgentService` in a new process resumes** — the agent cache is no longer required for resume.
`resume_agent`'s docstring is updated: durable when `LOTTIE_MESH_CHECKPOINT=sqlite` (the serve default);
in-memory/process-local otherwise.

The transport validates the body into a mesh-free `ResumeDecision` (§6); `resume_agent` converts it to
the engine's `ApprovalDecision` via a **lazy import inside the method** — `serve/service.py` is imported
by `serve/__init__`, so it must NOT import `lottie.mesh.schema` at module top (would break the base
install without `[mesh]`):

```python
        from lottie.mesh.schema import ApprovalDecision  # lazy: keep serve base-install mesh-free
        approval = ApprovalDecision(action=decision.action, edited_input=decision.edited_input)
        output = resume(thread_id, approval)
```

The REST transport modules (`rest_app`/`rest_schema`) never import mesh at all.

## 6. `POST /v1/agents/{name}/resume` — the route

Added to `rest_app.rest_routes` (a fourth route). Small request models in `rest_schema.py` —
**mesh-import-free** (so `rest_app`/`rest_schema` import without `[mesh]`):

```python
class ResumeDecision(BaseModel):
    action: Literal["approve", "reject"]
    edited_input: dict[str, str] = {}

class ResumeRequest(BaseModel):
    thread_id: str
    decision: ResumeDecision
```

`resume_agent` (in `serve/service.py`, which may import mesh) converts the `ResumeDecision` to the
engine's `ApprovalDecision`. `rest_schema.py` never imports `lottie.mesh.schema`.

Handler (mirrors `run_agent_route`'s shape):

```python
    async def resume_agent_route(request: Request) -> JSONResponse:
        name = request.path_params["name"]
        try:
            body = await request.json()
            req = ResumeRequest.model_validate(body)
        except (ValueError, ValidationError):
            return json_error(400, "invalid request body", type_="invalid_request")
        try:
            result = await anyio.to_thread.run_sync(
                lambda: svc.resume_agent(name, req.thread_id, req.decision)
            )
        except InputSecurityViolation:
            return json_error(400, "request blocked by content policy", type_="content_filter")
        except OutputSecurityViolation as exc:
            return JSONResponse(withheld_dict(name, input_tokens=exc.input_tokens, output_tokens=exc.output_tokens))
        except NotResumable:
            return json_error(400, f"agent '{name}' is not resumable", type_="not_resumable")
        except ThreadNotFound:
            return json_error(404, f"thread not found", type_="thread_not_found")
        except AgentNotFoundError:
            return json_error(404, f"agent '{name}' not found", type_="not_found")
        except (AgentLoadError, AgentExecutionError):
            return json_error(500, "internal error", type_="internal_error")
        return JSONResponse(run_result_dict(result))
```

`Route("/v1/agents/{name}/resume", resume_agent_route, methods=["POST"])`. The `ApprovalDecision` import
moves into `rest_schema.py` (it's `lottie.mesh.schema`, available with `[mesh]`; the REST module already
loads lazily so importing mesh schema at top is fine — but guard it: a base/`[api]`-without-`[mesh]`
install must still import `rest_app`. Use a lazy import of `ApprovalDecision` inside the handler/model
construction, or accept a plain `{action, edited_input}` dict and build `ApprovalDecision` lazily, so
`rest_app` import never hard-requires `[mesh]`).

> **Decision on the `[mesh]` coupling:** `ResumeRequest` must not hard-import `lottie.mesh.schema` at
> module top (would break `[api]`-without-`[mesh]`). Model `decision` as a small local
> `ResumeDecision(BaseModel){action: Literal["approve","reject"], edited_input: dict[str,str] = {}}` in
> `rest_schema.py` (no mesh import), and `resume_agent` (in `serve/service.py`, which may import mesh)
> converts it to the engine's `ApprovalDecision`. Keeps `rest_app`/`rest_schema` mesh-free.

## 7. Response & status taxonomy

Success → `run_result_dict(result)` — the same serialized `RunResult` as `/run`:
- resume that completes → `status="complete"`, `output` = the mesh `final`/history.
- resume that hits the **next** HITL gate → `status="interrupted"` + a new `thread_id` + `pending`
  (multi-gate meshes resume one gate at a time).
- output withheld → 200 `status="withheld"`, `output={}`, usage kept.

## 8. Inherited security & governance

`resume_agent` runs the output `SecurityGate` (now via `_check_output`, carrying metrics) and the resumed
mesh run goes through `BaseAgent.run` → audit/policy/cost. A test asserts a resume produces an audit
record. Input-gate: the decision body is scanned by the gate inside `run_agent`'s sibling path? — resume
does not re-run input sanitization on a decision (there's no agent Input); the decision is a typed
control message, validated by Pydantic. The output gate still applies. (Documented: resume's "input" is a
typed decision, not free content, so the input-injection gate is N/A; output withhold still enforced.)

## 9. Packaging & CLI

- **No new extra** — `[api]` (Starlette/uvicorn) + `[mesh]` (langgraph + langgraph-checkpoint-sqlite,
  already a dep) cover it. Durable resume needs `[mesh]`; without it a mesh agent can't be served anyway.
- `serve/__init__` still imports none of the HTTP modules.
- `cli/serve.py` sets `LOTTIE_MESH_CHECKPOINT=sqlite` (setdefault) in the `--port` branch.
- **CLAUDE.md:** note `/v1/agents/{name}/resume` + the `LOTTIE_MESH_CHECKPOINT` env in the serve docs.

## 10. Testing

All HTTP tests `pytest.importorskip("starlette")`-guarded; mesh tests `importorskip("langgraph")`.

- **`_resolve_checkpoint`** (unit): arg > env > "memory" precedence; `LOTTIE_MESH_CHECKPOINT=sqlite` →
  "sqlite"; unset → "memory".
- **`build_checkpointer("sqlite", root)`** (unit, `[mesh]`): returns a SqliteSaver; creates
  `<root>/.lottie/mesh/`. (Removes the `# pragma: no cover`.)
- **Engine `ThreadNotFoundError`** (integration, `[mesh]`): a sqlite-backed mesh, resume a bogus
  `thread_id` (both `approve` and `reject` decisions) → raises `ThreadNotFoundError` (never a raw
  langgraph/pydantic error).
- **Durable cross-process resume** (integration, `[mesh]`): run a mesh to interrupt with
  `checkpoint="sqlite"` + root; build a **fresh** `AgentService` (same root, new process simulated) and
  `resume_agent` the `thread_id` → succeeds. Proves rehydrate-by-thread_id, no shared in-memory state.
- **Service error mapping** (unit): non-mesh agent → `NotResumable`; unknown thread → `ThreadNotFound`;
  missing agent → `AgentNotFoundError`.
- **REST resume route** (integration, TestClient, `[mesh]`): build a mesh agent, `POST /run` →
  `status="interrupted"` + `thread_id`; `POST /resume` {thread_id, decision:approve} → 200 RunResult.
  Error cases: unknown agent → 404 `not_found`; non-mesh agent → 400 `not_resumable`; bogus thread → 404
  `thread_not_found`; bad body → 400 `invalid_request`. Output-withhold on resume → 200 `withheld`.
- **Governance on resume**: a resume writes an audit record (inherited path).
- **Base-install / `[api]`-without-`[mesh]`**: `import lottie.serve.rest_app` works without langgraph
  (no top-level mesh import); `build_http_app` builds; the resume route 500s/400s cleanly if invoked on a
  non-mesh, never ImportErrors at import.
- **Full gate**: `uv run pytest -q`, `uv run mypy --strict src`, `uv run ruff check` green; existing
  tests (incl. the OpenAI + REST suites and the `build_openai_app`/`build_http_app` APIs) unaffected.

## 11. Definition of done

`POST /v1/agents/{name}/resume` resumes an interrupted mesh from `{thread_id, decision}` → serialized
`RunResult` (complete / interrupted-again / withheld); errors per §6 with the new typed leaves
(`NotResumable` 400, `ThreadNotFound` 404, wrapped from the engine's `ThreadNotFoundError` — never a raw
langgraph/pydantic leak). Durable: `LangGraphEngine` resolves the checkpoint arg > `LOTTIE_MESH_CHECKPOINT`
env > `"memory"`; `lottie serve --port` sets `sqlite`; a fresh `AgentService` (same root) resumes a
checkpoint written by another — verified by a cross-process test. `build_checkpointer` sqlite path
exercised (no-cover removed). Security + audit/policy/cost inherited (no second gate);
`run_result_dict`/`withheld_dict`/`json_error` reused; `rest_app`/`rest_schema` stay `[mesh]`-import-free.
Residual limits documented (shared-fs requirement, concurrent-same-thread relies on langgraph versioning,
FU-6 msgpack). `uv run pytest -q` / `mypy --strict src` / `ruff check` green. Validate downstream in
lottie-lab Round 12 before merging. Commit on the feature branch; do not push until approved.
