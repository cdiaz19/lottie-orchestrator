# Phase 4 — OpenAI-Compatible `/v1/chat/completions` Transport — Design

> Stand up an OpenAI-compatible chat-completions endpoint over the existing serving core.
> A thin Starlette/uvicorn wrapper over `AgentService` — every agent that opts in (via a
> `chat:` config block) is reachable as an OpenAI "model". Non-streaming, single-shot this
> slice. Opt-in `[api]` extra, base install web-free. Reuses the existing fail-closed
> SecurityGate + audit/policy/cost path; adds no second gate.

- **Date:** 2026-06-18
- **Phase:** Phase 4 (integration), next slice after MCP stdio (Round 5, `serve/mcp_server.py`).
- **Branch:** `feat/openai-compat-transport` (off `main`).

---

## 1. Goal & scope

The serving core (`AgentService`) already lists and runs agents transport-agnostically, gated by a
fail-closed `SecurityGate`, with audit/policy/cost firing via `BaseAgent.run`. One transport exists:
MCP stdio (`serve/mcp_server.py`, a pure wrapper). This slice adds a second transport — an
**OpenAI-compatible `/v1/chat/completions` HTTP endpoint** — so any OpenAI client/SDK can call a
Lottie agent by pointing its base URL at `lottie serve --port`.

**In scope:**
- `POST /v1/chat/completions` (non-streaming) and `GET /v1/models`.
- A declared, opt-in **chat adapter** mapping a chat request to an agent's typed Pydantic Input.
- The shared Starlette ASGI app + `lottie serve --port` foundation (so a later generic-REST slice is
  a thin addition on the same app).
- An optional `[api]` extra; base install stays web-dependency-free.

**Out of scope (deferred, noted in §9):**
- Streaming (`stream: true`) — `BaseAgent` is sync-first; run-streaming is a carried deferral.
- Conversation memory / multi-turn — agents are stateless single-shot here.
- Generic REST (`/v1/agents/{name}/run`) — a follow-up slice on the same app.
- Auth (API keys), rate limiting, `/v1/completions` (legacy), embeddings.

**Locked decisions (resolved in brainstorming, do not relitigate):**
- **`model` = agent name** (mirrors MCP's one-typed-surface-per-agent).
- **messages → Input: declared chat adapter, opt-in.** An agent exposes itself on the chat endpoint
  only by declaring a `chat: {input_field, output_field}` block in `config.yaml`. Undeclared → not a
  "model" → 404. Keeps every Input fully typed (CLAUDE.md rule 2); no field-name guessing.
- **messages collapse: last `user` message only.** System + prior turns ignored; no user message → 400.
- **SecurityViolation: split.** Input-reject → HTTP 400 `content_filter`; output-withhold → HTTP 200
  with `finish_reason="content_filter"`, empty content, `usage` populated.
- **Stack: Starlette + uvicorn** (already in the dependency tree; FastAPI is absent and not added).
- **Non-streaming only this slice.**

## 2. The `[api]` extra & no-op posture

`pyproject.toml` gains an optional-dependency group:

```toml
api = ["starlette>=1.2.1", "uvicorn>=0.49"]
```

(versions pinned to the wheels already resolved in the venv: `starlette 1.2.1`, `uvicorn 0.49.0`).
`httpx` (already present) is the test client — a dev/test dependency, not a runtime one.

- **Base install pulls nothing new.** `serve/__init__.py` does NOT import `openai_app` (mirrors how it
  never imports `mcp_server`), so `import lottie.serve` works with neither `[serve]` nor `[api]`.
- **`serve/openai_app.py` imports Starlette at module top** → imported lazily (only by `lottie serve
  --port` and the skip-guarded tests). Absent extra → a friendly CLI error, never an import crash.
- All HTTP tests are `pytest.importorskip("starlette")`-guarded, so CI's base (no-extra) job stays green.

## 3. Config — the `chat:` block

`project/config.py` gains an optional nested model on `AgentConfig`:

```python
class ChatConfig(BaseModel):
    """Opt-in mapping that exposes an agent on the OpenAI chat endpoint."""
    input_field: str   # last user message content -> Input.<input_field>
    output_field: str  # Output.<output_field> -> assistant message content

class AgentConfig(BaseModel):
    ...
    chat: ChatConfig | None = None   # None -> agent not chat-exposed
```

Backward-compatible: existing `config.yaml` files (no `chat:` key) parse unchanged and expose nothing
on the chat endpoint. An agent is "chat-capable" iff `config.chat is not None`.

## 4. `serve/openai_app.py` — the transport

```python
def build_openai_app(root: Path, *, service: AgentService | None = None) -> Starlette:
    """Build a Starlette app exposing chat-capable agents over /v1/chat/completions."""
```

Pure wrapper over `AgentService` (constructs one if not injected). Two routes:

### `GET /v1/models`
Lists chat-capable agents (those whose `config.chat is not None`) as OpenAI model objects:
`{object:"list", data:[{id:<agent>, object:"model", created:<epoch>, owned_by:"lottie"}, ...]}`.
A broken/unloadable agent is skipped with a warning (same tolerance as MCP's `build_mcp_server`).

### `POST /v1/chat/completions`
1. Parse the body into a typed `ChatCompletionRequest` (§5). Malformed JSON / schema → 400
   `invalid_request_error`.
2. `stream is True` → 400 `invalid_request_error`, message "streaming is not supported".
3. Resolve the agent: `model` = agent name; load its config; **not chat-capable → 404
   `model_not_found`**.
4. Build the payload: take the **last `user`-role message**'s `content`; none present → 400
   `invalid_request_error` ("no user message"). `payload = {chat.input_field: content}`.
5. Run via the core, off the event loop:
   `result = await anyio.to_thread.run_sync(lambda: svc.run_agent(name, payload))`.
   `run_agent` does the **real** `Input.model_validate(payload)` — an Input needing more than
   `input_field` raises `InvalidInputError` → 400 (honest: that agent isn't single-field-chat-usable).
6. Map `result.output[chat.output_field]` → assistant content (`str(...)`); build the 200 response (§5).
7. Errors: map `ServeError` subtypes (§6).

`temperature`, `max_tokens`, `top_p`, etc. are accepted and **ignored** — the agent owns its provider
config; documented, not an error (standard for a compat shim).

## 5. Request / response schemas

New Pydantic models (`serve/openai_schema.py`, skip-guard-free — pure pydantic, no Starlette import):

```python
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    # sampling params accepted but ignored this slice:
    temperature: float | None = None
    max_tokens: int | None = None
    # (top_p, n, stop, ... tolerated via model_config extra="ignore")
```

Response (built as a plain dict / typed model, serialized by Starlette's `JSONResponse`):

```jsonc
{
  "id": "chatcmpl-<uuid4 hex>",
  "object": "chat.completion",
  "created": <int epoch seconds>,
  "model": "<agent name>",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "<output_field rendered>"},
    "finish_reason": "stop"            // "content_filter" on output-withhold
  }],
  "usage": {"prompt_tokens": <in>, "completion_tokens": <out>, "total_tokens": <in+out>},
  "lottie": {"latency_ms": <f>, "cost_usd": <f>, "status": "<status>"}  // non-standard extension
}
```

The `lottie` extension object carries the per-run metrics MCP surfaces as a `[lottie]` text block —
non-standard but harmless to OpenAI clients (extra keys are ignored). `id` uses `uuid.uuid4().hex`;
`created` uses `int(time.time())` (runtime-only; not in any deterministic/replay context).

## 6. Error mapping

OpenAI error envelope: `{"error": {"message": ..., "type": ..., "code": ..., "param": null}}`.
**Messages never echo the offending payload** (parity with the SecurityGate contract).

| Exception | HTTP | `type` | `code` |
|---|---|---|---|
| body parse / schema / `stream:true` / no user message | 400 | `invalid_request_error` | — |
| `AgentNotFoundError` / not chat-capable | 404 | `invalid_request_error` | `model_not_found` |
| `InvalidInputError` | 400 | `invalid_request_error` | — |
| `InputSecurityViolation` | 400 | `invalid_request_error` | `content_filter` |
| `AgentLoadError` / `AgentExecutionError` | 500 | `internal_error` | — |
| `OutputSecurityViolation` | **200** | — (see below) | — |

**`OutputSecurityViolation` is special:** the agent already ran, but the output is withheld. Return a
normal 200 chat-completion with `choices[0].message.content = ""`,
`choices[0].finish_reason = "content_filter"`, and `usage` populated from the run's metrics — the
OpenAI-authentic way a model refuses mid-generation.

### The metrics-on-withhold wrinkle

`AgentService.run_agent` runs the output gate *after* the agent run but raises before returning a
`RunResult`, so the transport would have no `usage` for the withheld response. Fix, keeping the single
chokepoint (no second gate, no duplicated orchestration):

- `serve/errors.py`: add two subclasses —
  ```python
  class InputSecurityViolation(SecurityViolation): ...
  class OutputSecurityViolation(SecurityViolation):
      def __init__(self, message: str, *, input_tokens: int = 0, output_tokens: int = 0) -> None: ...
  ```
  Both remain `SecurityViolation` ⊂ `ServeError`, so MCP (which maps any `ServeError` → `isError`) is
  unaffected.
- `security.py`: `check_input` raises `InputSecurityViolation`; `check_output` raises
  `OutputSecurityViolation` (gate stays metrics-free — it only sees the serialized string).
- `service.run_agent`: wrap the `check_output` call; on `OutputSecurityViolation`, re-raise it carrying
  `agent.last_metrics` tokens. The service owns the agent + metrics, so attaching them there (not in the
  gate) keeps responsibilities clean.

The OpenAI transport reads those token counts for `usage`; MCP ignores them. Existing serve-path tests
that assert "output gate trips → SecurityViolation" still hold (subtype is-a `SecurityViolation`).

## 7. CLI — `lottie serve --port`

`cli/serve.py` gains an optional `--port` / `-p`:

- **no `--port`** → stdio MCP (`serve_stdio`), unchanged.
- **`--port N`** → lazy-import `build_openai_app`; `uvicorn.run(app, host=..., port=N)`. On
  `ImportError`, raise a `typer.BadParameter` with `pip install lottie-orchestrator[api]` (mirrors the
  existing `[serve]` hint). `--host` defaults to `127.0.0.1`.

## 8. Inherited security & governance (no second gate)

The HTTP handler calls `AgentService.run_agent`, which already:
- runs the fail-closed `SecurityGate` (input sanitize+injection, output validate+secret), and
- runs the agent via `BaseAgent.run`, which fires policy → cost → audit → OTel.

`anyio.to_thread.run_sync` copies the current `contextvars` into the worker thread (the same
propagation property langgraph relies on, verified in Round 9), so a top-level HTTP run starts at audit
depth 0 → `root=True` and a root OTel span — correct. **A test injects a `SqliteAuditLogger`, POSTs, and
asserts a `root=True` audit record** — confirming audit/policy/cost fire on the HTTP path with no extra
wiring.

## 9. Out of scope / deferred (YAGNI)

- **Streaming** (`stream:true`) — returns 400 here; needs a sync→async run-streaming seam on
  `BaseAgent` (carried deferral). Its own slice.
- **Conversation memory / multi-turn** — only the last user message is used; system + history dropped.
  A conversational-agent surface (mapping the full `messages[]` into a chat-shaped Input) is future work.
- **Generic REST** (`/v1/agents/{name}/run`, arbitrary typed payload) — a thin follow-up slice on the
  same Starlette app this slice stands up.
- **Auth / API keys / rate limiting** — the endpoint binds `127.0.0.1` by default; network exposure +
  auth is a deployment slice.
- **Build-time chat-config validation** — an agent that declares `chat:` but whose Input needs more
  than `input_field` 400s at call time rather than being rejected at startup. Acceptable; a startup
  sanity check is a possible nicety, deferred.

## 10. Testing

All HTTP tests `pytest.importorskip("starlette")`-guarded; the Starlette `TestClient` (httpx) drives the
app in-process — no real socket/uvicorn needed.

- **Schemas** (unit): `ChatCompletionRequest` parse — valid body, extra sampling params ignored,
  missing `messages` → error.
- **Adapter** (unit): last-user-message extraction (skips system + assistant + prior user turns); no
  user message → the 400 path; payload shape `{input_field: content}`.
- **Happy path** (integration, MockLLM): a chat-capable agent → 200, `choices[0].message.content` =
  the `output_field`, `usage` tokens, `lottie` extension present, `finish_reason:"stop"`.
- **`GET /v1/models`**: lists only chat-capable agents; non-chat agents absent.
- **Errors**: unknown model → 404 `model_not_found`; undeclared-chat agent → 404; `stream:true` → 400;
  no user message → 400; an Input needing extra required fields → 400 `invalid_request_error`.
- **SecurityViolation split** (with a blocking gate): input-reject → 400 `content_filter`;
  output-withhold → 200, `finish_reason:"content_filter"`, empty content, `usage` populated. Assert no
  response message echoes the payload.
- **Governance on the HTTP path**: inject a `SqliteAuditLogger`, POST, assert one `root=True` audit
  record (and that a configured deny policy / over-budget yields the mapped error).
- **Base-install safety**: `import lottie.serve` and `lottie serve` (no `--port`) work with Starlette
  NOT installed; `lottie serve --port` without `[api]` → the friendly `[api]` hint.
- **Full gate at closeout**: `uv run pytest -q`, `uv run mypy --strict src`, `uv run ruff check` — all
  green; existing ~766 tests unaffected. Every test helper gets a return-type annotation (mypy --strict).

## 11. Definition of done

`POST /v1/chat/completions` serves any agent that declares a `chat:` block, mapping the last user
message → its typed Input and its `output_field` → assistant content, with OpenAI-shaped response +
`usage` + a `lottie` metrics extension; `GET /v1/models` lists chat-capable agents; `model` = agent
name. Error mapping per §6, including the split SecurityViolation (400 input / 200 `content_filter`
output with populated `usage`). `lottie serve --port` serves the app via uvicorn; no `--port` is
unchanged stdio MCP. Opt-in `[api]` extra; base install web-free; `serve/__init__` clean; all HTTP tests
skip-guarded. Security + audit/policy/cost inherited via `AgentService.run_agent` (no second gate),
confirmed by a `root=True` audit-on-HTTP test. `serve.openai_app` adds no import cycle. `uv run pytest
-q` / `mypy --strict src` / `ruff check` green. Validate downstream in lottie-lab Round 10 before
merging the PR. Commit on the feature branch; do not push until approved.
