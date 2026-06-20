# Real Token Streaming — Slice 3a: `BaseAgent.run_stream` core seam — Design

> The governance-preserving core seam for real token streaming: a streaming analog of `run()` that
> flows content deltas **through** `BaseAgent` so policy + cost + audit + usage all still fire — plus an
> opt-in `_stream` agent method and a usage-bearing provider streaming primitive. Core + `llm` layers
> ONLY. Nothing is exposed to clients (no SSE, no transport) — that is slice 3b.

- **Date:** 2026-06-20
- **Phase:** Phase 4+ — Real Token Streaming, slice 3a of 3b (provider `stream()` [#19] + `StreamingSecretGate`
  [#20] done → **this core seam** → 3b SSE wiring).
- **Branch:** `feat/real-token-streaming-3a` (off `main`).

---

## 1. Context & the non-negotiable

`stream:true` today is **format-level** (#18): run the agent fully, then chunk the finished output into SSE.
Real *token* streaming means yielding deltas as the LLM produces them. The locked constraint:

> **Content flows THROUGH `BaseAgent`, never around it.** A transport must NOT stream straight from the
> provider — that bypasses `BaseAgent.run`, the chokepoint where policy/cost/audit live, leaving streamed
> runs ungoverned and unaudited. REJECTED: transport-orchestrated passthrough. The agent exposes a
> streaming method; the transport consumes it.

So the agent grows a streaming path that mirrors `run()`'s governance + instrumentation, as a generator.

**Decomposition (locked):** 3a = the core seam **behind** the transport (this doc). 3b = the SSE wiring
(`StreamingSecretGate` over the deltas, the `anyio.to_thread` bridge, the real `/v1/chat/completions`
`stream:true` path). 3a ships with NO client-visible change; it can never ship to clients without 3b.

## 2. Grounding (confirmed against the code)

- **`InstrumentedRunnable.run`** (`core/runnable.py:54`): sets `_active_ctx = RunContext()`, times via
  `perf_counter`, runs `_execute`, and in `finally` calls `_record` (builds `RunMetrics` → `last_metrics`
  + `append_metrics`) and `span_set_metrics`, inside a `run_span`. The streaming seam must reproduce this
  template **as a generator** — `super().run()` is unusable (it calls the non-streaming `_execute` and
  returns a value).
- **`BaseAgent.run`** (`core/base_agent.py:77`): `self._policy.check()` then `self._cost.check()` (pre,
  with `_write_block` + re-raise on `PolicyViolation`/`BudgetExceeded`), then `_audit_depth.set(_depth()+1)`,
  `is_root = _depth()==1`, `super().run(data)`, `finally _write_audit(data, output, is_root)` + depth reset.
- **`complete()`** (`base_agent.py:152`): `self.llm.complete(...)` then `self._active_ctx.add_usage(usage,
  cost)`. **`LLMProvider.stream()` (slice 1) yields `str` deltas only — no usage** — so the streaming agent
  helper needs a *usage-bearing* provider primitive to keep audit/cost parity.
- **`RunContext.add_usage(usage: TokenUsage, cost_usd=0.0)`** (`core/metrics.py:51`).
- **`MockLLMProvider.complete`** returns zero `TokenUsage` — the usage-parity test therefore uses a small
  fixture provider returning known non-zero usage from BOTH `complete` and `stream_complete`.

## 3. The usage mechanism — generator return value (the one design addition)

The brief names `stream_complete` but leaves *how usage flows* open (slice-1 `stream()` is `str`-only).
Solution: a generator that **yields `str` deltas and `return`s the usage**, carried out-of-band in
`StopIteration.value` and captured by `yield from`. The delta stream stays pure text; usage rides alongside.

```python
# llm/base.py
class StreamResult(BaseModel):
    """Usage/cost for a streamed completion, delivered at stream end."""
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
```

`Generator[str, None, StreamResult]` is fully typed; `result = yield from gen` types `result: StreamResult`
under `mypy --strict`.

## 4. Provider layer — `LLMProvider.stream_complete`

Added alongside (not replacing) slice-1 `stream()`. `stream()` stays the content-only primitive (shipped,
tested, untouched); `stream_complete` is the **governed** primitive `run_stream` consumes — same `str`
deltas, plus usage at the end.

```python
# llm/base.py — concrete default on the ABC (no subclass breaks)
def stream_complete(
    self, messages: list[Message], model_params: Mapping[str, object] | None = None,
) -> Generator[str, None, StreamResult]:
    """Yield content deltas, returning usage/cost at stream end.

    Default: one-shot over complete() — yields the whole content in one delta, returns its usage."""
    response = self.complete(messages, model_params)
    yield response.content
    return StreamResult(usage=response.usage, cost_usd=response.cost_usd)
```

```python
# llm/litellm_provider.py — real streaming WITH final-chunk usage (the ONLY litellm streaming touchpoint, rule 1)
def stream_complete(self, messages, model_params=None) -> Generator[str, None, StreamResult]:
    params = dict(model_params or {})
    params.pop("stream", None); params.pop("stream_options", None)        # avoid kw collision (slice-1 lesson)
    payload = [{"role": m.role, "content": m.content} for m in messages]
    usage = TokenUsage(); cost = 0.0
    for chunk in litellm.completion(
        model=self._model, messages=payload, stream=True,
        stream_options={"include_usage": True}, **params,
    ):
        chunk_usage = getattr(chunk, "usage", None)                        # usage rides a final, choice-less chunk
        if chunk_usage is not None:
            usage = TokenUsage(input_tokens=chunk_usage.prompt_tokens or 0,
                               output_tokens=chunk_usage.completion_tokens or 0)
            cost = self._cost(chunk) or cost                               # same cost path as complete() (litellm_provider.py:63)
        if not chunk.choices:                                              # usage-only chunk → no delta (slice-1 guard)
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
    return StreamResult(usage=usage, cost_usd=cost)
```

> Cost/usage source = exactly what `complete()` uses: `self._cost(response)` →
> `litellm.completion_cost(completion_response=...)` (`litellm_provider.py:63`) and
> `response.usage.prompt_tokens/completion_tokens` (`:38`), applied here to the final usage-bearing chunk.
> If a particular litellm version omits cost on the streamed final chunk, `cost` stays `0.0` — an honest
> floor, never a crash (the `or cost` guard).

```python
# llm/mock.py — chunked replay sharing the queue with complete; zero usage (structurally correct)
def stream_complete(self, messages, model_params=None) -> Generator[str, None, StreamResult]:
    content = self._pop_response(messages)                                 # same eager queue advance as complete/stream
    for piece in re.findall(r"\S+\s*|\s+", content):
        yield piece
    return StreamResult()                                                  # MockLLM has no usage; parity tested via a fixture provider
```

## 5. Core — instrumentation generator (`InstrumentedRunnable`)

The streaming twin of `run()`, kept on `InstrumentedRunnable` so instrumentation stays in one place
(DRY; mirrors the existing instrumentation/governance split). Skills never call it.

```python
# core/runnable.py
def _instrument_stream(self, pieces: Iterator[str]) -> Iterator[str]:
    """run()'s instrumentation, streamed: time, accumulate usage (via the producer), record post."""
    ctx = RunContext(); self._active_ctx = ctx
    start = perf_counter(); success = True; error: str | None = None
    with run_span(self.name, self.kind) as span:
        try:
            yield from pieces                          # producer (_stream) calls stream_complete → ctx.add_usage
        except GeneratorExit:                          # consumer closed early — PARTIAL run, NOT clean success
            success = False; error = "stream closed before completion"
            raise                                      # must re-raise GeneratorExit
        except Exception as exc:
            success = False; error = repr(exc); span_set_error(span, exc); raise
        finally:
            self._record(ctx, start, success, error)   # writes last_metrics with usage accumulated so far
            span_set_metrics(span, self.last_metrics)
            self._active_ctx = None
```

- `_active_ctx` is set **before** iteration; `pieces = self._stream(data)` is a lazy generator whose body
  (which calls `stream_complete`) does not run until the first `next()`, by which point `_active_ctx` exists
  — so usage accumulates correctly.
- **`GeneratorExit` is `BaseException`, not `Exception`** — `run()`'s `except Exception` would miss it, so it
  is handled explicitly: mark the run not-success with a fixed error label, then re-raise (a generator must
  let `GeneratorExit` propagate). `_record` only *writes* (never yields) — safe during close, unlike the
  slice-2 flush.

## 6. Core — `BaseAgent` streaming API

```python
# core/base_agent.py
class NotStreamable(RuntimeError):
    """Raised if _stream is invoked on an agent that did not opt in (capability check is the guard)."""

def _stream(self, data: InputT) -> Iterator[str]:           # OPT-IN: default raises; override to enable
    raise NotStreamable(f"{self.name} does not implement _stream")

@classmethod
def supports_streaming(cls) -> bool:                        # explicit override check (NOT an exception path)
    return cls._stream is not BaseAgent._stream

def stream_complete(                                        # agent analog of complete(): usage → _active_ctx
    self, messages: list[Message], model_params: Mapping[str, object] | None = None,
) -> Iterator[str]:
    result = yield from self.llm.stream_complete(messages, model_params)
    if self._active_ctx is not None:
        self._active_ctx.add_usage(result.usage, result.cost_usd)

def run_stream(self, data: InputT) -> Iterator[str]:
    """Streaming analog of run(): same policy/cost pre-gates, instrumented stream, audit post."""
    self._pre_run_gates(data)                              # policy+cost; _write_block + raise on block
    token = _audit_depth.set(_depth() + 1)
    is_root = _depth() == 1
    try:
        yield from self._instrument_stream(self._stream(data))
    finally:
        try:
            self._write_audit(data, None, is_root)         # output=None → output_sha256=None (deltas, no single Output)
        finally:
            _audit_depth.reset(token)
```

**`_pre_run_gates` (refactor, shared with `run`):** the existing pre-check block in `run()` (policy→cost,
`_write_block` + re-raise on `PolicyViolation`/`PolicyEscalation`/`BudgetExceeded`) is extracted verbatim
into `_pre_run_gates(data)` and called by both `run()` and `run_stream()` — DRY, one gate definition.

- `run_stream` is a **generator** (it `yield`s), so `_pre_run_gates` runs on the **first `next()`** — before
  any piece is produced. A policy deny / over-budget therefore raises *before* the first delta, exactly the
  "blocks before any piece" semantics. The transport pulls the first delta inside its `try`, mapping a raise
  to an error before the SSE body starts.
- **Audit on stream end:** `_write_audit` reads `last_metrics` (set by `_instrument_stream._record`), so a
  clean stream writes `status="ok"` with full usage; an early close writes `status="error"`,
  `error="stream closed before completion"`, with partial usage. We deliberately reuse the existing
  `ok`/`error` `AuditRecord` status (no new `interrupted` status — avoids schema churn); the error label
  records the partial/interrupted nature. `output_sha256` is `None` (a stream has no single typed Output;
  output integrity is the serve-layer gate's concern in 3b).

## 7. Governance layering (explicit — mirrors where gates already live)

- **CORE (`run_stream`):** policy + cost (pre) + audit (post) + usage accumulation. `run_stream` does **NOT**
  run the output security gate.
- **SERVE (3b):** the input gate + the slice-2 `StreamingSecretGate` wrapping the output deltas
  incrementally. The output gate is already serve-path-only today; the streaming secret gate belongs at the
  transport boundary, exactly like the non-streaming one.

## 8. Capability & the holistic-validator carve-out

The transport (3b) selects real-stream vs format-fallback via `agent.supports_streaming()` — an explicit
override check, never a caught exception. Agents that don't opt in (including any whose output gate needs
whole-document validation and so *cannot* truly stream) simply don't implement `_stream` → they fall back to
the slice-1 format-level path in 3b. One consistent rule. No production agent opts in within 3a; the opt-in
is exercised by a small streaming test agent (3a has no transport).

## 9. Testing (3a — pure unit, MockLLM streaming + a usage fixture; NO transport)

`core/tests/` + `llm/tests/`. No real LLM (rule 5).

- **`run_stream` streams incrementally** — a `_StreamingAgent(BaseAgent)` whose `_stream` does
  `yield from self.stream_complete([...])` over a multi-word `MockLLMProvider` → `list(agent.run_stream(d))`
  has `len > 1` and `"".join(...)` reconstructs the canned response.
- **Policy deny blocks before any piece** — attach a denying `PolicyGate`; `next(agent.run_stream(d))` raises
  `PolicyViolation`, zero pieces yielded, a `denied` block row is audited.
- **Cost over-budget blocks pre-run** — a `CostGate` over budget; first `next()` raises `BudgetExceeded`,
  zero pieces, a `budget_exceeded` block row audited.
- **Audit post (clean)** — drain `run_stream`; one `ok` audit row with the run's usage.
- **Audit post (early close → PARTIAL)** — pull one piece, `.close()` the generator; one `error` row with
  `error="stream closed before completion"` and partial (not full) usage; assert the held tail was not
  produced.
- **Usage parity** — a fixture `LLMProvider` returning `TokenUsage(input=11, output=7)` + a cost from BOTH
  `complete` and `stream_complete`; assert `run(d).last_metrics` and the drained `run_stream(d).last_metrics`
  carry equal `input_tokens` / `output_tokens` / `cost_usd`.
- **`supports_streaming`** — `True` for `_StreamingAgent`, `False` for a plain `BaseAgent` subclass (default
  `_stream`); calling the default `_stream` raises `NotStreamable`.
- **Provider `stream_complete` default fallback** — a minimal `complete`-only provider → one delta == full
  content, and the captured `StreamResult.usage` equals `complete`'s usage.
- **`MockLLMProvider.stream_complete`** — multi-delta reconstruct-exact; shares the queue with `complete`
  (index advances; exhaustion raises); returns a zero-usage `StreamResult`.
- **`LiteLLMProvider.stream_complete`** (`litellm.completion` monkeypatched) — fake chunks with content
  deltas + a final usage-only (choice-less) chunk → yields the deltas (empties/None skipped, no IndexError on
  the choice-less chunk) and returns a `StreamResult` with the final-chunk usage; assert `stream=True` and
  `stream_options={"include_usage": True}` were passed.
- **Full gate** — `uv run pytest -q`, `uv run mypy --strict src`, `uv run ruff check` green; existing suite
  unaffected (additive; `complete`/`stream`/`run` behavior unchanged apart from the `_pre_run_gates` extract,
  which is covered by the existing policy/cost block tests).

## 10. Files

- **`src/lottie/llm/base.py`** — add `StreamResult` model + `stream_complete` concrete default.
- **`src/lottie/llm/litellm_provider.py`** — `stream_complete` override (final-chunk usage; only litellm
  streaming touchpoint).
- **`src/lottie/llm/mock.py`** — `stream_complete` override (chunked replay + zero-usage `StreamResult`).
- **`src/lottie/core/runnable.py`** — `_instrument_stream` generator.
- **`src/lottie/core/base_agent.py`** — `NotStreamable`, `_stream` default, `supports_streaming`,
  `stream_complete`, `run_stream`; extract `_pre_run_gates` (shared with `run`).
- **Tests** — `src/lottie/llm/tests/` (provider) + `src/lottie/core/tests/` (agent seam).

No web deps; no `serve/` change. `serve/__init__` stays clean.

## 11. Out of scope (slice 3b)

- `AgentService` streaming method; the `StreamingSecretGate` over the output deltas; the `anyio.to_thread`
  sync→async bridge; the real `/v1/chat/completions` `stream:true` path replacing format-fallback for opt-in
  agents; format-fallback selection in the transport.
- **OutputValidation** over a stream (oversized running-byte cap; empty-at-flush → error finish).
- A lab round — 3a is behind the transport; the real-streaming round lands with 3b.

## 12. Definition of done

`BaseAgent.run_stream(data) -> Iterator[str]` streams content deltas through the governance chokepoint:
`_pre_run_gates` (policy+cost, shared with `run`) fire before the first delta; `_instrument_stream` times the
stream, accumulates usage into `_active_ctx` via `stream_complete`, and records `last_metrics` post (PARTIAL
on early `GeneratorExit`); `_write_audit` writes the row in `finally`. `LLMProvider.stream_complete` yields
`str` deltas and returns `StreamResult` usage (default one-shot; `LiteLLMProvider` real final-chunk usage;
`MockLLMProvider` chunked replay). `supports_streaming()` is an explicit override check. No SSE / transport /
async / output-gate wiring (3b). `uv run pytest -q` / `mypy --strict src` / `ruff check` green. Commit on the
feature branch; do not push until approved.
