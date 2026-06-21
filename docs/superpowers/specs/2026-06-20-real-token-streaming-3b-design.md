# Real Token Streaming — Slice 3b: SSE seam — Design

> Wires slice-3a's governed `BaseAgent.run_stream` to the wire: a real-token `/v1/chat/completions`
> `stream:true` path that streams LLM deltas through the slice-2 `StreamingSecretGate` to SSE, bridging the
> sync generators to the async response via `anyio.to_thread`. Capability-gated: only agents that opted into
> `_stream` stream for real; everyone else keeps the existing format-level SSE. Serve layer only.

- **Date:** 2026-06-20
- **Phase:** Phase 4+ — Real Token Streaming, slice 3b of 3b (3a core seam [merged #21] → **this SSE seam**).
- **Branch:** `feat/real-token-streaming-3b` (off `main`).

---

## 1. Context

3a (merged, #21) shipped `BaseAgent.run_stream(data) -> Generator[str, None, None]` — governed token
streaming through the chokepoint (policy + cost + audit + usage), opt-in via `_stream`, with
`supports_streaming()` as the explicit capability check. Slice 2 (#20) shipped `StreamingSecretGate.scan_stream`
— a sync generator that line-buffers deltas and yields only scanned-clean text, raising
`OutputSecurityViolation` on a secret. This slice connects them to the HTTP transport.

Today (`serve/openai_app.py`) `stream:true` is **format-level**: `svc.run_agent` runs the agent fully, then
`_stream_response` chunks the finished string into SSE. 3b replaces that with REAL streaming **for opt-in
agents**, keeping format-level as the fallback for everyone else.

**Locked architecture (3a brief, do not relitigate):** content flows THROUGH `run_stream` (never straight
from the provider); the gate wraps the deltas at the transport boundary (exactly like the non-stream output
gate); capability detection is `supports_streaming()`, not an exception path; the sync→async bridge is
`anyio.to_thread` (the MCP pattern).

## 2. Grounding (confirmed against the code)

- `serve/openai_app.py::chat_completions` — parses, resolves the chat config (`_chat_config` → 404
  `model_not_found`), maps the last user message to `{input_field: content}`, runs `svc.run_agent` off-loop
  via `anyio.to_thread.run_sync`, maps `ServeError`s, and on `req.stream` calls `_stream_response` (format-level).
- `serve/service.py::AgentService` — `run_agent` does: `_require_agent` → `_gate.check_input(json.dumps(payload))`
  → `load_input_model` + `model_validate` → `_get_agent` (cached) → `agent.run` → `_check_output` → `RunResult`.
  `_get_agent(name, provider)` get-or-builds + caches.
- `serve/security.py::SecurityGate` — `check_input` (sanitize + injection) / `check_output`
  (OutputValidation + secret). The streaming path reuses `check_input` verbatim; the streaming OUTPUT gate is
  the slice-2 `StreamingSecretGate` (secret-only, incremental) — **OutputValidation over a stream is deferred**
  (slice-2/3a scope note; oversized/empty cap is future work).
- `serve/stream_gate.py::StreamingSecretGate.scan_stream(deltas) -> Iterator[str]` — sync, raises
  `OutputSecurityViolation`.
- `serve/openai_schema.py::chat_completion_chunks(agent, content, finish_reason) -> list[str]` — builds the SSE
  events for a COMPLETE content (role chunk, one content chunk, finish chunk, `[DONE]`). 3b needs a per-delta
  encoder.
- MCP one-shot bridge: `await anyio.to_thread.run_sync(lambda: svc.run_agent(...))`. The streaming bridge
  repeats `await anyio.to_thread.run_sync(next, sync_gen)`.

## 3. Service — `AgentService.stream_agent`

One new method; the gate instance is added to `__init__`.

```python
# __init__: self._stream_gate = StreamingSecretGate()   # secret-only incremental output gate

def stream_agent(
    self, name: str, payload: Mapping[str, object], *, provider: str | None = None,
) -> Iterator[str] | None:
    """Real-token stream of an opt-in agent's output, secret-gated incrementally.

    Returns None if the agent does not implement _stream — the transport then uses the format-level
    fallback. Otherwise gates the input + validates EAGERLY (so input/validation errors raise here,
    before the SSE starts), then returns scan_stream(run_stream(data)) — a lazy sync generator whose
    governance (policy/cost/audit) fires when the transport pulls it.
    """
    self._require_agent(name)                                  # AgentNotFoundError
    agent = self._get_agent(name, provider)                    # AgentLoadError
    if not agent.supports_streaming():
        return None                                            # -> caller: format fallback (no gating yet)
    self._gate.check_input(json.dumps(payload))                # InputSecurityViolation (pre-stream)
    try:
        input_model = load_input_model(self._root, name)
    except Exception as exc:                                   # noqa: BLE001
        raise AgentLoadError(f"cannot load agent '{name}': {exc}") from exc
    try:
        data = input_model.model_validate(payload)
    except ValidationError as exc:
        raise InvalidInputError(f"invalid input for '{name}': {exc}") from exc
    return self._stream_gate.scan_stream(agent.run_stream(data))
```

- **Capability check runs BEFORE gating** — a non-streamable agent returns `None` without paying the input
  gate, so the fallback path (which re-runs `run_agent`, re-gating) is not double-gated on the common case.
- **Input gate + validation are eager** (run at call time, before the generator is returned) → they raise as
  `ServeError`s the transport maps to pre-stream HTTP status (400/404/500). `run_stream` itself is lazy: its
  policy/cost pre-gates + the LLM call do not fire until the transport pulls the first item.
- **No `_check_output`** — the full output gate (OutputValidation + whole-text secret) is replaced on the
  streaming path by the incremental `StreamingSecretGate`. OutputValidation-over-a-stream is deferred.

## 4. Transport — the real SSE path

`chat_completions` gains a streaming branch BEFORE the format-level one. The decision is capability-driven via
the `None` sentinel from `stream_agent`.

```python
if req.stream:
    try:
        gen = await anyio.to_thread.run_sync(lambda: svc.stream_agent(req.model, payload))
    except InputSecurityViolation:
        return json_error(400, "request blocked by content policy",
                          type_="invalid_request_error", code="content_filter")
    except InvalidInputError:
        return json_error(400, f"input does not fit model '{req.model}'", type_="invalid_request_error")
    except AgentNotFoundError:
        return _model_not_found(req.model)
    except (AgentLoadError, AgentExecutionError):
        return json_error(500, "internal error", type_="internal_error")
    if gen is not None:
        return StreamingResponse(_sse_real(req.model, gen), media_type="text/event-stream")
    # gen is None -> fall through to the existing format-level path below
```

The existing non-streaming + format-fallback code stays. (Format fallback re-runs `svc.run_agent`, which
re-gates the input — acceptable; the capability check returned before the streaming gate ran.)

### 4.1 Error philosophy (NO priming — matches the brief's literal error spec)

`stream:true` on a streamable agent commits to a **200 SSE** as soon as the body starts. Therefore:

| When | Error | Result |
|---|---|---|
| Pre-stream (eager in `stream_agent`) | input reject | **400** `content_filter` |
| Pre-stream | invalid input | **400** |
| Pre-stream | agent not found | **404** `model_not_found` |
| Pre-stream | load failure | **500** |
| In-stream (lazy, during iteration) | clean completion | SSE `finish_reason: "stop"` |
| In-stream | secret detected | SSE `finish_reason: "content_filter"` — prior clean lines stay, the secret line is NEVER yielded |
| In-stream | policy deny / over budget / agent raised | SSE `finish_reason: "error"` |
| In-stream | client disconnect | bridge closes the generator → `run_stream` records a PARTIAL audit row |

Rationale: policy/cost/secret fire lazily inside `run_stream`/`scan_stream`, after the 200 SSE has begun, so
they cannot become an HTTP status — they surface as the terminal `finish_reason`. This deliberately does NOT
prime the first line to convert them to pre-stream statuses: priming would need the agent handle for
withhold-usage reporting (coupling the transport to `last_metrics`) for marginal benefit. Non-stream parity is
preserved where it matters (input reject is pre-stream both ways; a secret still never leaves). A streamed
policy/cost denial surfaces as `finish_reason: "error"` rather than the non-stream 500 — an accepted,
documented difference (you cannot send a 500 after a 200).

**No `usage` object in the SSE** this slice (OpenAI's streamed usage rides a final
`stream_options.include_usage` chunk — the format-level path doesn't emit it either; deferred). The audit
ledger still records full/partial usage server-side via 3a.

### 4.2 The async bridge

```python
_STREAM_DONE = object()  # module-level sentinel: StopIteration across the thread boundary

def _safe_next(gen: Iterator[str]) -> object:
    try:
        return next(gen)
    except StopIteration:
        return _STREAM_DONE

async def _sse_real(model: str, gen: Iterator[str]) -> AsyncIterator[str]:
    """Bridge a sync (run_stream -> scan_stream) generator to SSE, pulling each delta off the event loop."""
    enc = ChatChunkEncoder(model)
    yield enc.role()
    finish = "stop"
    try:
        while True:
            item = await anyio.to_thread.run_sync(_safe_next, gen)
            if item is _STREAM_DONE:
                break
            yield enc.content(item)            # item is a scanned-clean line (str)
    except OutputSecurityViolation:
        finish = "content_filter"              # a secret tripped scan_stream; the secret line was never yielded
    except Exception:                          # noqa: BLE001 — policy/cost/agent failure mid-stream
        finish = "error"
    finally:
        gen.close()                            # disconnect/secret/error -> run_stream PARTIAL audit (close is a no-op if exhausted)
    yield enc.finish(finish)
    yield enc.done()
```

- `_safe_next` converts `StopIteration` to a sentinel so it survives the `to_thread` boundary (a raw
  `StopIteration` crossing an `await` would become a `RuntimeError`).
- `gen.close()` in `finally` propagates `GeneratorExit` into `run_stream` → its 3a PARTIAL-audit path fires on
  early disconnect. On normal exhaustion the generator is already closed; `.close()` is a no-op.
- `item` is always a `str` (a scanned-clean complete line). `mypy` needs a cast/assert after the sentinel
  check: `if item is _STREAM_DONE: break` narrows the rest to `str` via an `assert isinstance(item, str)` or
  by typing `_safe_next -> str | object` and asserting — the implementer picks the minimal mypy-clean form.

## 5. SSE encoder — `openai_schema.py`

A pure, stateful encoder sharing one id/created across a streamed response (no Starlette import → unit-testable
without `[api]`):

```python
class ChatChunkEncoder:
    """Builds OpenAI chat.completion.chunk SSE events sharing one id/created (for a real-token stream)."""

    def __init__(self, model: str) -> None:
        self._model = model
        self._id = f"chatcmpl-{uuid.uuid4().hex}"
        self._created = int(time.time())

    def _event(self, delta: dict[str, object], finish: str | None) -> str:
        body = {"id": self._id, "object": "chat.completion.chunk", "created": self._created,
                "model": self._model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        return f"data: {json.dumps(body)}\n\n"

    def role(self) -> str: return self._event({"role": "assistant"}, None)
    def content(self, text: str) -> str: return self._event({"content": text}, None)
    def finish(self, reason: str) -> str: return self._event({}, reason)
    @staticmethod
    def done() -> str: return "data: [DONE]\n\n"
```

`chat_completion_chunks` (the format-fallback builder) is **refactored to build on the encoder** (DRY — same
event shapes): role, then `content(content)` only if non-empty, then `finish(finish_reason)`, then `done()`.
Its existing signature + output bytes are unchanged (existing tests stay green).

## 6. Layering / packaging

- All changes in `serve/` (`service.py`, `openai_app.py`, `openai_schema.py`). No `core`/`llm` change.
- `StreamingSecretGate` already lives in `serve/`; `service.py` imports it (serve→serve). No new dependency;
  `[api]` extra unchanged; `serve/__init__` still imports neither `openai_app` nor `http_app`.
- The single `AgentService` chokepoint is preserved — `stream_agent` reuses `_get_agent`, `_require_agent`,
  `_gate`, so audit/policy/cost/input-gate all still fire on the streaming HTTP path (via `run_stream`).

## 7. Testing

`serve/tests/` (Starlette `TestClient`; `MockLLMProvider` streaming; no real LLM). The opt-in test agent
declares a `chat:` block and overrides `_stream`.

- **Service `stream_agent` returns None for a non-streamable agent** → transport uses fallback.
- **Service `stream_agent` gates input eagerly** → a poisoned payload raises `InputSecurityViolation` at call
  time (before any generator is pulled).
- **Real SSE happy path** — opt-in agent + `stream:true` → multiple `chat.completion.chunk` events whose
  concatenated `delta.content` equals the agent's output; a `role` chunk first, a `finish_reason:"stop"`
  chunk, then `[DONE]`. Assert MORE THAN ONE content chunk (proves real chunking, not format-level).
- **Secret mid-stream** — an agent whose `_stream` emits clean lines then a line containing `AKIA…` →
  the SSE delivers the prior clean chunks, ends `finish_reason:"content_filter"`, and the secret string
  appears in NONE of the emitted bytes.
- **Non-streamable agent + `stream:true`** → format-level fallback over SSE (single content chunk; existing
  behavior unchanged).
- **Input reject is pre-stream** — a poisoned payload with `stream:true` → **400** `content_filter` JSON (not
  an SSE).
- **Governance still fires** — a policy-denied (or over-budget) opt-in agent streamed → SSE ends
  `finish_reason:"error"`, and an audit row was written (the run went through `run_stream`).
- **Encoder unit tests** — `ChatChunkEncoder` events share one id; `role`/`content`/`finish`/`done` shapes;
  `chat_completion_chunks` byte-output unchanged after the refactor.
- **Bridge order** — deltas arrive in order (sync→async preserves sequence).
- **Full gate** — `uv run pytest -q`, `uv run mypy --strict src`, `uv run ruff check` green.

## 8. Definition of done

`/v1/chat/completions` with `stream:true` on an agent that opted into `_stream` streams real LLM deltas through
`run_stream` → `StreamingSecretGate` → SSE, deltas pulled off the event loop via `anyio.to_thread`; a secret
ends the stream `content_filter` with no leak and the prior clean chunks retained; a non-streamable agent and
every non-stream request keep their current behavior; input rejection is a pre-stream 400; client disconnect
records a PARTIAL audit row. `AgentService.stream_agent` returns `Iterator[str] | None` (None → fallback),
gating input + validating eagerly and reusing the one chokepoint. `serve/__init__` stays web-free. Gate green.
A lab Round (real streaming end-to-end — the round deferred from slices 1–3a) validates this AFTER merge.
Commit on the feature branch; do not push until approved.
```
