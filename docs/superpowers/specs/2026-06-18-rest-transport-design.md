# Phase 4 — Generic REST Transport — Design

> Add a Lottie-native REST surface — `GET /v1/agents`, `GET /v1/agents/{name}`,
> `POST /v1/agents/{name}/run` — on the SAME Starlette app the OpenAI-compat slice stands up,
> served by `lottie serve --port`. Any agent is callable by its real typed Pydantic Input (no
> chat adapter); the run returns the full `RunResult`. Reuses the existing fail-closed
> SecurityGate + audit/policy/cost path — no second gate.

- **Date:** 2026-06-18
- **Phase:** Phase 4 (integration), follow-up to the OpenAI-compat slice (PR #15, on `main`).
- **Branch:** `feat/rest-transport` (off `main`).

---

## 1. Goal & scope

The OpenAI-compat slice (`serve/openai_app.py`) exposes only **chat-capable** agents (those that
declare a `chat:` block) under an OpenAI-shaped contract. This slice adds the **Lottie-native REST
surface** the OpenAI slice's design doc (§9) deferred: every agent reachable by its **actual typed
Input**, returning the full `RunResult` (output + metrics + run status). It composes onto the same
Starlette app and the same `lottie serve --port` server.

**In scope:**
- `GET /v1/agents` — list ALL agents (name + provider).
- `GET /v1/agents/{name}` — agent detail incl. the Input JSON schema (so a client knows the payload).
- `POST /v1/agents/{name}/run` — run an agent from its raw typed Input JSON → serialized `RunResult`.
- App composition: route-provider functions (`openai_routes`, `rest_routes`) assembled by a new
  `build_http_app(root)`; `lottie serve --port` serves that.

**Out of scope (deferred):**
- **Resume** (`POST /v1/agents/{name}/resume`) — the `run` response SURFACES a mesh interrupt
  (`status="interrupted"` + `thread_id` + `pending`), but acting on it is a follow-up slice
  (stateful; the in-memory checkpointer is process-local, FU-9).
- **Streaming** — `BaseAgent` is sync-first (carried deferral; same as the OpenAI slice).
- Auth / rate limiting / pagination.

**Locked decisions (resolved in brainstorming, do not relitigate):**
- **App composition = route-provider functions composed in one app.** `openai_app.py` →
  `openai_routes(svc, root)` + `build_openai_app`; `rest_app.py` → `rest_routes(svc, root)` +
  `build_rest_app`; `http_app.py` → `build_http_app(root)` (one `AgentService`,
  `Starlette(routes=[*openai_routes, *rest_routes])`). CLI serves `build_http_app`.
- **Endpoints = run + list + detail(schema); resume deferred** (run still surfaces interrupt state).
- **Output-withhold = HTTP 200 with the output stripped**, `status="withheld"`, `usage`/metrics
  preserved (carried on `OutputSecurityViolation` — plumbing already built in the OpenAI slice).
  Consistent with the OpenAI slice's 200-`content_filter` posture: the run executed; the body is
  stripped, not an error.
- **No second gate** — reuse `AgentService.run_agent` → `BaseAgent.run`; audit/policy/cost/security
  all fire on the REST path.
- **No chat adapter** — the request body IS the agent's typed Input; `run_agent` does the real
  `model_validate`.

## 2. App composition refactor

### `serve/openai_app.py` (refactor, behavior-preserving)
Extract the two route handlers into a route-provider function; `build_openai_app` delegates so the
existing public API (used by the OpenAI tests + the lab Round-10 driver) is unchanged:

```python
def openai_routes(svc: AgentService, root: Path) -> list[Route]:
    """The /v1/models + /v1/chat/completions routes, closed over svc + root."""
    # (the existing list_models / chat_completions handlers move here, unchanged)
    return [
        Route("/v1/models", list_models, methods=["GET"]),
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
    ]

def build_openai_app(root: Path, *, service: AgentService | None = None) -> Starlette:
    svc = service or AgentService(root)
    return Starlette(routes=openai_routes(svc, root))
```

### `serve/rest_app.py` (new)
```python
def rest_routes(svc: AgentService, root: Path) -> list[Route]:
    """The /v1/agents[...] Lottie-native REST routes, closed over svc + root."""
    return [
        Route("/v1/agents", list_agents, methods=["GET"]),
        Route("/v1/agents/{name}", agent_detail, methods=["GET"]),
        Route("/v1/agents/{name}/run", run_agent_route, methods=["POST"]),
    ]

def build_rest_app(root: Path, *, service: AgentService | None = None) -> Starlette:
    svc = service or AgentService(root)
    return Starlette(routes=rest_routes(svc, root))
```
`build_rest_app` exists so the REST routes are testable in isolation; production serves the combined
app.

### `serve/http_app.py` (new — the production entrypoint)
```python
def build_http_app(root: Path) -> Starlette:
    """One AgentService over both transports — the app `lottie serve --port` serves."""
    svc = AgentService(root)
    return Starlette(routes=[*openai_routes(svc, root), *rest_routes(svc, root)])
```
One shared `AgentService` ⇒ one chokepoint; both route groups inherit the same gate + governance.
`rest_app.py` is the seam: REST is now a `*rest_routes(svc, root)` line in the compose list.

All three modules import Starlette at top → imported lazily only; `serve/__init__.py` imports NONE of
them, so `import lottie.serve` stays web-free.

## 3. Endpoints

### `GET /v1/agents`
Lists every discovered agent (NOT just chat-capable):
```json
{"agents": [{"name": "digest", "provider": "anthropic/claude-sonnet-4-6"}, ...]}
```
Built from `AgentService.list_agents()` (`AgentInfo` → name + provider). Import-free; a broken agent
still lists (parity with `list_agents`).

### `GET /v1/agents/{name}`
```json
{"name": "digest", "provider": "anthropic/...", "input_schema": { ...JSON Schema... }}
```
`input_schema` = `load_input_model(root, name).model_json_schema()`. Agent missing → 404 `not_found`;
schema unloadable (broken agent) → 500 `internal_error` (honest: the agent exists but can't introspect).

### `POST /v1/agents/{name}/run`
Body = the agent's **typed Input JSON** (e.g. `{"query": "..."}` for `DigestAgentInput`). Flow:
1. Agent missing (`agents/{name}/agent.py` absent) → 404 `not_found`.
2. Parse body as JSON (must be a JSON object) → else 400 `invalid_request`.
3. `result = await anyio.to_thread.run_sync(lambda: svc.run_agent(name, body))` — `run_agent` does
   the real `Input.model_validate(body)`.
4. Map errors (§4); success → serialized `RunResult` (§5).

## 4. Error mapping

Reuses `serve/error_map.py::json_error(status, message, *, type_, code=None)` (the OpenAI envelope
`{"error": {message, type, code, param}}` — `code`/`param` null for REST). REST-native `type_`
strings. **Messages never echo request/response payload** (only the agent name, which is not payload).

| Case | HTTP | `type_` |
|---|---|---|
| body not a JSON object / `InvalidInputError` | 400 | `invalid_request` |
| `InputSecurityViolation` | 400 | `content_filter` |
| agent not found (`AgentNotFoundError`) | 404 | `not_found` |
| `AgentLoadError` / `AgentExecutionError` | 500 | `internal_error` |
| `OutputSecurityViolation` | **200** | *(not an error — see §5)* |

`run_agent`'s `AgentNotFoundError` is belt-and-suspenders: the route pre-checks the agent file and
404s first; kept for safety.

## 5. Success & withhold responses

### 200 — normal run
The serialized `RunResult` (a Lottie-native shape, NOT OpenAI):
```json
{
  "agent": "digest",
  "output": { ...the agent's Output.model_dump()... },
  "status": "complete",
  "latency_ms": 12.0,
  "input_tokens": 8,
  "output_tokens": 5,
  "cost_usd": 0.0,
  "thread_id": null,
  "pending": null
}
```
A **mesh** agent that interrupts returns the same shape with `status="interrupted"`, a `thread_id`,
and a `pending` object — surfaced so a future resume slice (or a client) can act; this slice does not
resume.

### 200 — output withheld
When the OUTPUT gate trips, the agent already ran. Return the RunResult shape with the body stripped:
```json
{
  "agent": "digest", "output": {}, "status": "withheld",
  "latency_ms": 0.0, "input_tokens": 4, "output_tokens": 6,
  "cost_usd": 0.0, "thread_id": null, "pending": null
}
```
`input_tokens`/`output_tokens` come from the `OutputSecurityViolation` the service raises (it carries
the run metrics). The withheld output is NEVER placed in the response. This mirrors the OpenAI slice's
200-`content_filter`: a filtered success, not an error, with usage reported.

A small builder in `serve/rest_schema.py` keeps the handler thin and unit-testable without Starlette:
```python
def run_result_dict(result: RunResult) -> dict[str, object]: ...      # the 200 normal shape
def withheld_dict(agent: str, *, input_tokens: int, output_tokens: int) -> dict[str, object]: ...
def agent_list_dict(infos: list[AgentInfo]) -> dict[str, object]: ...
def agent_detail_dict(name: str, provider: str | None, input_schema: dict[str, object]) -> dict[str, object]: ...
```
(`RunResult` and `AgentInfo` already exist in `serve/schema.py`.)

## 6. Inherited security & governance (no second gate)

The REST `run` handler calls `AgentService.run_agent`, which already runs the fail-closed `SecurityGate`
(input sanitize+injection, output validate+secret) and the agent via `BaseAgent.run` (policy → cost →
audit → otel). `anyio.to_thread.run_sync` copies contextvars into the worker thread (the Round-9 /
`e99d42e` property), so a top-level REST run is `root=True` in audit. **A test injects a
`SqliteAuditLogger`, POSTs to `/v1/agents/{name}/run`, and asserts a `root=True` record** — confirming
governance fires on the REST path identically to the chat path.

## 7. Packaging & CLI

- **No new extra** — `[api]` already pins `starlette` + `uvicorn`.
- `serve/__init__.py` imports none of `http_app`/`rest_app`/`openai_app`; `import lottie.serve` stays
  starlette-free (verified by a test).
- `cli/serve.py`: the `--port` branch lazy-imports **`build_http_app`** (was `build_openai_app`) and
  passes it to `uvicorn.run`. The `[api]` ImportError hint is unchanged. No `--port` is still stdio MCP.
- **CLAUDE.md:** reconcile the `lottie serve` docs to note that `--port` now also serves the REST
  endpoints (`/v1/agents[...]`) alongside the OpenAI ones.

## 8. Testing

All HTTP tests `pytest.importorskip("starlette")`-guarded; the Starlette `TestClient` drives the apps
in-process.

- **rest_schema builders** (unit, no Starlette): `run_result_dict` shape from a `RunResult`;
  `withheld_dict` (output `{}`, status `withheld`, tokens); `agent_list_dict`; `agent_detail_dict`.
- **`GET /v1/agents`**: lists all scaffolded agents with provider.
- **`GET /v1/agents/{name}`**: returns `input_schema` (a JSON Schema with the Input's fields); unknown
  agent → 404 `not_found`.
- **`POST .../run` happy path** (MockLLM): typed body → 200, `output` = the agent Output, `status`
  `complete`, metrics present.
- **Errors**: unknown agent → 404; body that fails Input validation → 400 `invalid_request`;
  non-object body → 400.
- **Security**: injection in the typed body → 400 `content_filter` (no payload echo); a secret in the
  output → 200 `status="withheld"`, `output == {}`, usage present, secret absent from the body.
- **Governance on REST**: inject a `SqliteAuditLogger`, POST `/run`, assert a `root=True` record.
- **Composition**: `build_http_app(root)` serves BOTH — one `/v1/models` (OpenAI) call and one
  `/v1/agents` (REST) call both succeed against the single app.
- **CLI**: `lottie serve --port N` hands `build_http_app(root)` to `uvicorn.run` (monkeypatched);
  no `--port` still stdio.
- **Base-install safety**: `import lottie.serve` works with Starlette absent; `serve/__init__` clean.
- **Full gate**: `uv run pytest -q`, `uv run mypy --strict src`, `uv run ruff check` green; the
  existing OpenAI tests + lab-facing `build_openai_app` API unchanged.

## 9. Definition of done

`build_http_app` serves the OpenAI routes AND the REST routes (`GET /v1/agents`, `GET
/v1/agents/{name}`, `POST /v1/agents/{name}/run`) over one `AgentService`; `lottie serve --port` serves
it. A run takes the agent's raw typed Input and returns the serialized `RunResult` (mesh interrupt
surfaced via `status`/`thread_id`/`pending`); errors per §4; output-withhold is a 200 with the body
stripped and `status="withheld"` + usage. Security + audit/policy/cost inherited via
`AgentService.run_agent` (no second gate), confirmed by a `root=True` audit-on-REST test. `openai_app`'s
public API (`build_openai_app`) is unchanged; `serve.rest_app`/`http_app` add no import cycle; base
install web-free; all HTTP tests skip-guarded. `uv run pytest -q` / `mypy --strict src` / `ruff check`
green. Validate downstream in lottie-lab Round 11 before merging. Commit on the feature branch; do not
push until approved.
