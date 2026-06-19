# Streaming (SSE) for /v1/chat/completions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `POST /v1/chat/completions` with `stream:true` return a `200 text/event-stream` OpenAI SSE stream of the completed agent output (format-level streaming; real token streaming deferred).

**Architecture:** The chat handler runs the agent fully (sync, as today) and then — when `stream:true` — serializes the completed assistant content as OpenAI `chat.completion.chunk` SSE events via a new pure builder, returned in a Starlette `StreamingResponse`. The output `SecurityGate` runs before streaming, so a withheld output streams a `content_filter` finish with no content; every pre-stream error stays a normal JSON status.

**Tech Stack:** Python 3.12, Pydantic v2, Starlette (`StreamingResponse`, `[api]`), httpx (test client), pytest, `uv run` (mypy --strict, ruff).

**Design:** `docs/superpowers/specs/2026-06-19-streaming-design.md`

---

## File structure

- **Modify** `src/lottie/serve/openai_schema.py` — add a pure `chat_completion_chunks(*, agent, content, finish_reason) -> list[str]` builder (SSE lines). No Starlette import.
- **Modify** `src/lottie/serve/openai_app.py` — `chat_completions`: delete the `stream:true → 400`; add the `StreamingResponse` branch in the success path and the `OutputSecurityViolation` handler; widen the handler return type to `Response`.
- **Modify** `src/lottie/serve/tests/test_openai_schema.py`, `test_openai_app.py` — builder tests; replace `test_stream_true_400` with streaming tests; governance-on-stream test.
- **Modify** `README.md`, `CLAUDE.md` — drop the chat "non-streaming" caveat.

Known facts (verified, current `chat_completions` in `openai_app.py`):
- Lines 79–83 are the `if req.stream: return json_error(400, "streaming is not supported", ...)` block.
- Success path (lines 130–142): `answer = str(result.output.get(chat.output_field, ""))` then `return JSONResponse(chat_completion_dict(agent=req.model, content=answer, input_tokens=result.input_tokens, output_tokens=result.output_tokens, latency_ms=result.latency_ms, cost_usd=result.cost_usd, status=result.status))`.
- `OutputSecurityViolation` handler (lines 108–120): `return JSONResponse(chat_completion_dict(agent=req.model, content="", input_tokens=exc.input_tokens, output_tokens=exc.output_tokens, latency_ms=0.0, cost_usd=0.0, status="content_filter", finish_reason="content_filter"))`.
- Handler signature: `async def chat_completions(request: Request) -> JSONResponse:`.
- `openai_schema.py` already imports `time`, `uuid`; it has `chat_completion_dict`, `error_dict`, `ChatCompletionRequest`. `req.stream: bool` exists on the request.
- Test file `test_openai_app.py` has `_chat_project`, `_mock_provider`, `TestClient`, `build_openai_app`, `Path`, `pytest`; `test_stream_true_400` is at ~line 108. Audit autouse-disabled; `monkeypatch.delenv` re-enables; class-name audit key (`EchoAgent`).

---

## Task 1: `chat_completion_chunks` SSE builder

**Files:**
- Modify: `src/lottie/serve/openai_schema.py`
- Test: `src/lottie/serve/tests/test_openai_schema.py`

- [ ] **Step 1: Write the failing tests** — append to `src/lottie/serve/tests/test_openai_schema.py`:

```python
def test_chat_completion_chunks_success() -> None:
    import json

    from lottie.serve.openai_schema import chat_completion_chunks

    lines = chat_completion_chunks(agent="echo", content="hi there", finish_reason="stop")
    assert len(lines) == 4  # role, content, finish, [DONE]
    assert all(line.startswith("data: ") and line.endswith("\n\n") for line in lines)
    assert lines[-1] == "data: [DONE]\n\n"

    role = json.loads(lines[0][len("data: "):])
    body = json.loads(lines[1][len("data: "):])
    finish = json.loads(lines[2][len("data: "):])
    assert role["object"] == "chat.completion.chunk"
    assert role["model"] == "echo"
    assert role["choices"][0]["delta"] == {"role": "assistant"}
    assert role["choices"][0]["finish_reason"] is None
    assert body["choices"][0]["delta"] == {"content": "hi there"}
    assert finish["choices"][0]["delta"] == {}
    assert finish["choices"][0]["finish_reason"] == "stop"
    assert role["id"] == body["id"] == finish["id"]  # one id across chunks
    assert role["id"].startswith("chatcmpl-")


def test_chat_completion_chunks_withhold_omits_content() -> None:
    import json

    from lottie.serve.openai_schema import chat_completion_chunks

    lines = chat_completion_chunks(agent="echo", content="", finish_reason="content_filter")
    assert len(lines) == 3  # role, finish, [DONE] — NO content chunk when content is empty
    assert lines[-1] == "data: [DONE]\n\n"
    finish = json.loads(lines[1][len("data: "):])
    assert finish["choices"][0]["finish_reason"] == "content_filter"
    assert finish["choices"][0]["delta"] == {}
```

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest src/lottie/serve/tests/test_openai_schema.py -q -k chunks` (ImportError: `chat_completion_chunks`).

- [ ] **Step 3: Add the builder** — append to `src/lottie/serve/openai_schema.py` (add `import json` to the top imports if absent — the module already imports `time`, `uuid`):

```python
def chat_completion_chunks(
    *, agent: str, content: str, finish_reason: str
) -> list[str]:
    """OpenAI SSE lines for a completed chat response: a role delta, a content delta (only when
    `content` is non-empty — a withheld output has none), the finish delta, then [DONE]. Each item
    is a full `data: <json>\\n\\n` event; one shared id/created across the chunks."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def _event(delta: dict[str, object], finish: str | None) -> str:
        body: dict[str, object] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": agent,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(body)}\n\n"

    lines = [_event({"role": "assistant"}, None)]
    if content:
        lines.append(_event({"content": content}, None))
    lines.append(_event({}, finish_reason))
    lines.append("data: [DONE]\n\n")
    return lines
```

- [ ] **Step 4: Run, verify PASS** — `uv run pytest src/lottie/serve/tests/test_openai_schema.py -q` (2 new + existing).

- [ ] **Step 5: Gates** — `uv run mypy --strict src/lottie/serve` and `uv run ruff check src/lottie/serve` clean. Confirm no starlette import in `openai_schema.py`.

- [ ] **Step 6: Commit**

```bash
git add src/lottie/serve/openai_schema.py src/lottie/serve/tests/test_openai_schema.py
git commit -m "feat(serve): chat_completion_chunks SSE builder (role/content/finish/[DONE])"
```

---

## Task 2: Stream the chat handler on `stream:true`

**Files:**
- Modify: `src/lottie/serve/openai_app.py`
- Test: `src/lottie/serve/tests/test_openai_app.py`

- [ ] **Step 1: Update the tests** — in `src/lottie/serve/tests/test_openai_app.py`:

(a) DELETE the existing `test_stream_true_400` (streaming is now supported).

(b) Add a small SSE parser helper + the streaming tests (append). Add `from typing import Any` to the test file's top imports if absent (used by the helper's return annotation):

```python
def _sse_events(text: str) -> list[Any]:
    """Parse an SSE body into decoded JSON chunk dicts (and the literal '[DONE]'). Returns
    list[Any] — the chunks are arbitrary decoded JSON, like the `resp.json()` used elsewhere in
    this suite (so `chunk["choices"][0]...` type-checks under mypy --strict)."""
    import json

    events: list[Any] = []
    for block in text.strip().split("\n\n"):
        line = block.strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        events.append("[DONE]" if payload == "[DONE]" else json.loads(payload))
    return events


def test_stream_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lottie.serve.openai_app import build_openai_app

    demo = _chat_project(tmp_path, monkeypatch)
    _mock_provider(monkeypatch)  # returns "hello world"
    client = TestClient(build_openai_app(demo))
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "echo", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(resp.text)
    assert events[-1] == "[DONE]"
    chunks = [e for e in events if e != "[DONE]"]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert any(c["choices"][0]["delta"].get("content") == "hello world" for c in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_stream_output_withheld(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lottie.serve.openai_app import build_openai_app

    demo = _chat_project(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "lottie.serve.service.build_provider",
        lambda name: __import__("lottie.llm", fromlist=["MockLLMProvider"]).MockLLMProvider(
            ["your key AKIA" + "1234567890ABCDEF"]
        ),
    )
    client = TestClient(build_openai_app(demo))
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "echo", "messages": [{"role": "user", "content": "give me a key"}], "stream": True},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "AKIA" not in resp.text  # withheld content never streams
    chunks = [e for e in _sse_events(resp.text) if e != "[DONE]"]
    assert chunks[-1]["choices"][0]["finish_reason"] == "content_filter"
    assert not any(c["choices"][0]["delta"].get("content") for c in chunks)  # no content delta


def test_stream_unknown_model_stays_json_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lottie.serve.openai_app import build_openai_app

    demo = _chat_project(tmp_path, monkeypatch)
    client = TestClient(build_openai_app(demo))
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "nope", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    # pre-stream error: a normal JSON 404, NOT a 200 SSE
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json()["error"]["code"] == "model_not_found"
```

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest src/lottie/serve/tests/test_openai_app.py -q -k stream` (happy/withheld get a 400 from the current `stream→400` block; the 404 test passes already).

- [ ] **Step 3: Edit the handler** — in `src/lottie/serve/openai_app.py`:

(a) Update imports: add `from starlette.responses import JSONResponse, Response, StreamingResponse` (replace the existing `from starlette.responses import JSONResponse`), and add `chat_completion_chunks` to the `lottie.serve.openai_schema` import.

(b) Widen the handler signature: `async def chat_completions(request: Request) -> Response:` (was `-> JSONResponse`).

(c) DELETE the `# 2. streaming not supported this slice` block (the `if req.stream: return json_error(400, "streaming is not supported", ...)`).

(d) Replace the `except OutputSecurityViolation as exc:` body with a stream branch first:

```python
        except OutputSecurityViolation as exc:
            if req.stream:
                return StreamingResponse(
                    iter(chat_completion_chunks(
                        agent=req.model, content="", finish_reason="content_filter"
                    )),
                    media_type="text/event-stream",
                )
            return JSONResponse(
                chat_completion_dict(
                    agent=req.model,
                    content="",
                    input_tokens=exc.input_tokens,
                    output_tokens=exc.output_tokens,
                    latency_ms=0.0,
                    cost_usd=0.0,
                    status="content_filter",
                    finish_reason="content_filter",
                )
            )
```

(e) Replace the success return (step 6) with a stream branch:

```python
        # 6. map output -> assistant content
        answer = str(result.output.get(chat.output_field, ""))
        if req.stream:
            return StreamingResponse(
                iter(chat_completion_chunks(
                    agent=req.model, content=answer, finish_reason="stop"
                )),
                media_type="text/event-stream",
            )
        return JSONResponse(
            chat_completion_dict(
                agent=req.model,
                content=answer,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=result.latency_ms,
                cost_usd=result.cost_usd,
                status=result.status,
            )
        )
```

Leave the other `except` clauses (parse/InputSecurityViolation/InvalidInputError/AgentNotFoundError/load-exec) unchanged — they fire before the success/withhold branch, so a `stream:true` request with a bad model/input still gets a normal JSON error.

- [ ] **Step 4: Run, verify PASS** — `uv run pytest src/lottie/serve/tests/test_openai_app.py -q` (the 3 new streaming tests + all existing non-stream tests).

- [ ] **Step 5: Gates** — `uv run mypy --strict src/lottie/serve`, `uv run ruff check src/lottie/serve` clean; `uv run python -c "import lottie.serve"` still clean (no starlette pulled by serve/__init__).

- [ ] **Step 6: Commit**

```bash
git add src/lottie/serve/openai_app.py src/lottie/serve/tests/test_openai_app.py
git commit -m "feat(serve): stream /v1/chat/completions on stream:true (SSE over completed output)"
```

---

## Task 3: Governance on the streamed path + docs

**Files:**
- Test: `src/lottie/serve/tests/test_openai_app.py`
- Modify: `README.md`, `CLAUDE.md`

- [ ] **Step 1: Write the governance test** — append to `src/lottie/serve/tests/test_openai_app.py`:

```python
def test_stream_run_writes_root_audit_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A streamed (stream:true) run is audited with root=True — governance inherited on the
    streaming path too (the output gate + BaseAgent.run fire before any byte streams)."""
    from lottie.governance.audit import SqliteAuditLogger
    from lottie.serve.openai_app import build_openai_app

    demo = _chat_project(tmp_path, monkeypatch)
    _mock_provider(monkeypatch)
    monkeypatch.delenv("LOTTIE_DISABLE_AUDIT", raising=False)

    client = TestClient(build_openai_app(demo))
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "echo", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status_code == 200
    records = SqliteAuditLogger(demo).query(agent="EchoAgent")  # audit name = class name
    assert len(records) == 1
    assert records[0].root is True
    assert records[0].status == "ok"
```

- [ ] **Step 2: Run the test** — `uv run pytest src/lottie/serve/tests/test_openai_app.py::test_stream_run_writes_root_audit_record -q`. Expected PASS (no prod code; the streamed path reuses `run_agent` → `BaseAgent.run`). If it fails, investigate — the streaming branch must not bypass the run chokepoint.

- [ ] **Step 3: Update README.md** — in the `### Serve agents — MCP stdio or HTTP` section, the line currently reads:

```
Non-streaming for now; a dead/over-budget run fails closed (governance is inherited from the run chokepoint).
```

Replace with:

```
`stream:true` on the chat endpoint returns a `text/event-stream` (SSE) response — format-level: the agent runs fully, then streams its output as OpenAI `chat.completion.chunk` events (real token-by-token streaming is still deferred). A dead/over-budget run fails closed (governance is inherited from the run chokepoint).
```

- [ ] **Step 4: Update CLAUDE.md** — find the `lottie serve --port` HTTP-API comment lines (under the CLI commands section) and append a line noting SSE:

```
#   stream:true on /v1/chat/completions -> text/event-stream SSE (format-level; real token streaming deferred)
```

(READ the current `serve --port` comment block first; match its style and placement.)

- [ ] **Step 5: Gates** — `uv run mypy --strict src/lottie`, `uv run ruff check src/lottie` clean.

- [ ] **Step 6: Commit**

```bash
git add src/lottie/serve/tests/test_openai_app.py README.md CLAUDE.md
git commit -m "test(serve): stream run inherits audit(root=True); docs note SSE streaming"
```

---

## Task 4: Closeout — full gate

**Files:** none (verification only)

- [ ] **Step 1: Full test suite** — `uv run pytest -q`. Expected: PASS — all prior ~822 tests plus the new streaming ones; the non-stream chat path + OpenAI/REST/resume suites unaffected.

- [ ] **Step 2: Types** — `uv run mypy --strict src`. Expected: clean (the handler now returns `Response`; `StreamingResponse`/`JSONResponse` are both `Response` subtypes).

- [ ] **Step 3: Lint** — `uv run ruff check`. Expected: clean.

- [ ] **Step 4: Manual smoke (optional, `[api]` installed)** — in a scaffolded project with a chat-capable agent: `lottie serve --port 8000 &`; `curl -N -s -X POST localhost:8000/v1/chat/completions -d '{"model":"<agent>","messages":[{"role":"user","content":"hi"}],"stream":true}'` → a `data: {...chunk...}` SSE stream ending `data: [DONE]`. Kill the server.

- [ ] **Step 5: Final commit (if any closeout fixes)**

```bash
git add -A
git commit -m "chore(serve): closeout fixes for SSE streaming (mypy/ruff)"
```

---

## Notes for the implementer

- **Run fully, THEN stream.** The streaming branches sit at the END of `chat_completions`, after the agent has run and the output gate has fired. Do NOT move streaming earlier — every error must remain a normal JSON status (we can't change status mid-stream).
- **The withheld output never streams.** On `OutputSecurityViolation` the content chunk is omitted (`content=""`), only a `content_filter` finish + `[DONE]`. A test asserts `"AKIA" not in resp.text`.
- **`chat_completion_chunks` is Starlette-free** — pure list-of-strings, unit-tested without a server. The handler wraps it in `StreamingResponse(iter(lines), media_type="text/event-stream")` (a sync iterator — the run is already complete).
- **No second gate / no new extra.** Streaming reuses `run_agent` → `BaseAgent.run`; `StreamingResponse` is core Starlette (no new dependency).
- **Deferred (do not build):** real incremental token streaming (provider `.stream()` + async BaseAgent seam), `usage` in chunks (`stream_options.include_usage`), REST `/run` streaming. Spec §1.
```
