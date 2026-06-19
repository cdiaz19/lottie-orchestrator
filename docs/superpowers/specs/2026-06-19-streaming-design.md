# Phase 4 — Streaming (SSE) for `/v1/chat/completions` — Design

> Honor `stream:true` on the OpenAI-compatible chat endpoint: run the agent fully (sync, as today),
> then emit the result as an OpenAI **Server-Sent-Events** stream (`text/event-stream`). This is
> **format-level** streaming — it makes OpenAI clients that require `stream:true` work, with the
> correct SSE wire format — not real token-by-token generation (deferred). The fail-closed
> SecurityGate still runs **before** any byte streams, so a withheld output is never sent.

- **Date:** 2026-06-19
- **Phase:** Phase 4 (integration), slice 5 — after OpenAI-compat (#15), REST (#16), durable resume
  (#17), all on `main`.
- **Branch:** `feat/streaming` (off `main`).

---

## 1. Goal & scope

Every prior HTTP slice deferred `stream:true` because `BaseAgent` is sync-first: `LLMProvider` exposes
only `complete()` (no streaming), and `_execute` returns a complete typed Output. Today the chat handler
returns `400 "streaming is not supported"`. This slice makes `stream:true` work at the **wire-format
level**: run the agent fully, then serialize the completed assistant content as an OpenAI SSE stream
(`chat.completion.chunk` events). That unblocks the large population of OpenAI clients/SDKs that *require*
`stream:true` and parse SSE — without the BaseAgent/provider rearchitecture real streaming needs.

**In scope:**
- `POST /v1/chat/completions` with `stream:true` → `200 text/event-stream`, OpenAI SSE chunks of the
  completed output (single content delta).
- The output `SecurityGate` runs before streaming; a withheld output streams a `content_filter` finish
  with no content.

**Out of scope (deferred):**
- **Real incremental token streaming** — needs a new `LLMProvider.stream()` (litellm streaming) + an
  async/streaming seam through `BaseAgent`, and reworks the output gate (a streamed secret can't be
  un-sent — gate per-chunk or buffer). Its own slice spanning `llm`/`core`/`serve`.
- **`usage` in chunks** — OpenAI only includes it when `stream_options.include_usage` is set; not
  modeled here.
- **REST `/run` streaming** — REST clients consume the full `RunResult` JSON; SSE there is non-standard.
- Word/token-chunked deltas — the completed content is emitted as a **single** content delta (honest:
  no faked incrementalism over already-computed content).

**Locked decisions (resolved in brainstorming, do not relitigate):**
- **Format-level streaming over the completed output** (run fully sync, then stream) — NOT real token
  streaming.
- **Single content chunk.**
- **OpenAI `/v1/chat/completions` only.**
- **Output gate before streaming** — withhold → `content_filter` finish, content never streamed.
- Pre-stream errors stay normal JSON (the run completes before the SSE response opens).

## 2. Flow — run fully, THEN stream

The chat handler already does: parse body → `stream` check → resolve chat-capable model → last-user
adapter → `run_agent` off the event loop (with the error mapping) → output gate → response. This slice
changes only the **terminal** step and removes the early `stream:true` → 400.

Pipeline order (unchanged up to the response):
1. Parse `ChatCompletionRequest`; malformed → 400 (JSON).
2. **(removed)** the `if req.stream: return 400` early-return.
3. Resolve the chat model; not chat-capable → 404 `model_not_found` (JSON).
4. Last user message → payload; none → 400 (JSON).
5. `result = await anyio.to_thread.run_sync(lambda: svc.run_agent(model, payload))` with the existing
   except mapping — `InputSecurityViolation`→400, `InvalidInputError`→400, `AgentNotFoundError`→404,
   `(AgentLoadError, AgentExecutionError)`→500 — all **JSON** (still pre-stream).
6. **Terminal branch:**
   - Success: `if req.stream:` → `StreamingResponse(_sse(...success chunks...), media_type="text/event-stream")`; else the existing `JSONResponse(chat_completion_dict(...))`.
   - `OutputSecurityViolation`: `if req.stream:` → `StreamingResponse(...content_filter chunks...)`; else the existing 200 withheld JSON.

Because the agent runs to completion before step 6, the SSE response opens only on success/withhold, and
every error is a normal JSON status — a client gets a real 4xx/5xx, never a half-open stream.

## 3. SSE chunk format

OpenAI `chat.completion.chunk` events, one shared `id = "chatcmpl-<uuid4 hex>"` and `created = int(epoch)`
across all chunks of a response. Each event is a line `data: <json>\n\n`; the stream ends with
`data: [DONE]\n\n`.

```jsonc
// chunk 1 — role
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":<t>,"model":"<agent>",
       "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

// chunk 2 — content (omitted entirely when content is empty, e.g. withhold)
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":<t>,"model":"<agent>",
       "choices":[{"index":0,"delta":{"content":"<full output>"},"finish_reason":null}]}

// chunk 3 — finish
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":<t>,"model":"<agent>",
       "choices":[{"index":0,"delta":{},"finish_reason":"<stop|content_filter>"}]}

data: [DONE]
```

`object` is `"chat.completion.chunk"` (distinct from the non-stream `"chat.completion"`). No `usage`
field (see §1 deferred).

## 4. Output withhold over SSE

The output gate runs in step 5 (inside `run_agent` → `_check_output`), so a withhold is an
`OutputSecurityViolation` raised before step 6 — i.e. before any byte streams. The handler's
`except OutputSecurityViolation` branch, when `req.stream`, returns a `StreamingResponse` whose chunks are
`chat_completion_chunks(agent, content="", finish_reason="content_filter")` → role chunk, **no content
chunk** (content is empty), finish chunk with `content_filter`, `[DONE]`. The withheld content is never
placed in any chunk (parity with the non-stream 200 `content_filter`). Tokens spent are not reported (no
`usage` in the stream this slice).

## 5. Builders — `openai_schema.py`

A pure, Starlette-free helper (unit-testable):

```python
def chat_completion_chunks(
    *, agent: str, content: str, finish_reason: str
) -> list[str]:
    """Build the OpenAI SSE lines for a completed chat response: role delta, a content delta
    (only when `content` is non-empty), the finish delta, then [DONE]. Each line is a full
    `data: <json>\\n\\n` SSE event. One shared id/created across the chunks."""
```

Returns `list[str]` (each a `data: …\n\n` string, last is `data: [DONE]\n\n`). Builds the same
`id`/`created` once and reuses across chunks. A small private `_chunk(id_, created, agent, delta,
finish_reason)` builds one `chat.completion.chunk` dict; the public fn json-encodes each into an SSE line.
The handler wraps the list: `StreamingResponse(iter(lines), media_type="text/event-stream")` (the run is
already complete, so a synchronous iterator is correct — no real async streaming).

(`chat_completion_dict` / `error_dict` from the OpenAI slice are unchanged and still used by the
non-stream path.)

## 6. The handler — `openai_app.py`

Inside `chat_completions` (in `openai_routes`):
- **Delete** the `if req.stream: return json_error(400, "streaming is not supported", ...)` block (step 2).
- Add the streaming imports: `from starlette.responses import StreamingResponse` and
  `from lottie.serve.openai_schema import chat_completion_chunks`.
- In the success path, replace the single `return JSONResponse(chat_completion_dict(...))` with a branch:
  ```python
      answer = str(result.output.get(chat.output_field, ""))
      if req.stream:
          return StreamingResponse(
              iter(chat_completion_chunks(agent=req.model, content=answer, finish_reason="stop")),
              media_type="text/event-stream",
          )
      return JSONResponse(chat_completion_dict(agent=req.model, content=answer, ...))
  ```
- In the `except OutputSecurityViolation as exc` branch, add the stream variant:
  ```python
      except OutputSecurityViolation as exc:
          if req.stream:
              return StreamingResponse(
                  iter(chat_completion_chunks(agent=req.model, content="", finish_reason="content_filter")),
                  media_type="text/event-stream",
              )
          return JSONResponse(chat_completion_dict(agent=req.model, content="", ..., finish_reason="content_filter"))
  ```

`StreamingResponse` defaults to HTTP 200. All other `except` clauses (input security, invalid input,
not-found, load/exec) are unchanged JSON errors — reached before the success/withhold branch, so they fire
for `stream:true` too (a streamed request with a bad model still gets a 404 JSON).

## 7. Inherited security & governance

Unchanged: the streamed request goes through the same `run_agent` → `SecurityGate` + `BaseAgent.run`
(policy → cost → audit → otel) chokepoint. The output gate runs before any byte streams. A test asserts a
`stream:true` run produces a `root=True` audit record (governance inherited on the streaming path too).

## 8. Packaging

No new extra (`[api]` already covers Starlette; `StreamingResponse` is core Starlette). `serve/__init__`
unchanged; base install stays web-free. No CLI change (`lottie serve --port` already serves the app).
CLAUDE.md/README: drop the "non-streaming" caveat for the chat endpoint; note `stream:true` returns SSE
(format-level; real token streaming still deferred).

## 9. Testing

All HTTP tests `pytest.importorskip("starlette")`-guarded; Starlette `TestClient` drives the app.

- **`chat_completion_chunks`** (unit, no Starlette): success (`content="hi"`, `finish_reason="stop"`) →
  3 SSE data lines + `[DONE]`; role chunk first, content chunk present, finish chunk with `stop`; each
  line starts `data: ` and ends `\n\n`; `object=="chat.completion.chunk"`; shared `id`. Withhold
  (`content=""`, `finish_reason="content_filter"`) → role + finish + `[DONE]`, **no content chunk**.
- **Streaming happy path** (integration, MockLLM): `POST` with `stream:true` → 200,
  `content-type` startswith `text/event-stream`; parse `resp.text` → assert a `delta.role=="assistant"`
  chunk, a `delta.content == "<output>"` chunk, a `finish_reason=="stop"` chunk, and `data: [DONE]`.
- **Streaming withhold** (integration, AKIA secret in output): 200 `text/event-stream`; a
  `finish_reason=="content_filter"` chunk, NO content delta, `[DONE]`, and `"AKIA"` NOT in `resp.text`.
- **Pre-stream error stays JSON**: `stream:true` with an unknown model → **404** JSON
  `model_not_found` (NOT an SSE 200); `stream:true` with a non-object/invalid body → 400 JSON.
- **Governance on the streamed path**: inject a `SqliteAuditLogger`, `POST` `stream:true`, assert one
  `root=True` audit record.
- **Non-stream unaffected**: the existing `chat_completions` JSON tests still pass; the old
  `stream:true → 400` test (if present) is replaced by the streaming tests.
- **Full gate**: `uv run pytest -q`, `uv run mypy --strict src`, `uv run ruff check` green; existing
  ~822 tests unaffected.

## 10. Definition of done

`POST /v1/chat/completions` with `stream:true` returns `200 text/event-stream` — role + single content +
finish (`stop`) chunks + `[DONE]` — of the fully-run agent's output; a withheld output streams a
`content_filter` finish with no content (never the withheld bytes); every pre-stream failure is a normal
JSON 4xx/5xx; the non-stream path is unchanged. Security + audit/policy/cost inherited (output gate before
streaming), confirmed by a `root=True` audit-on-stream test. `chat_completion_chunks` is Starlette-free
and unit-tested; base install web-free. Real token streaming, `usage`-in-chunks, and REST streaming are
documented as deferred. `uv run pytest -q` / `mypy --strict src` / `ruff check` green. Validate downstream
in lottie-lab Round 13 before merging. Commit on the feature branch; do not push until approved.
