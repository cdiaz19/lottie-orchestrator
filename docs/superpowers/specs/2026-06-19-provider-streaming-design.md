# Real Token Streaming — Slice 1: `LLMProvider.stream()` — Design

> The provider-layer seam for real token streaming: add `LLMProvider.stream()` yielding assistant
> content deltas. A concrete default (one-shot over `complete()`) keeps every existing provider working;
> `LiteLLMProvider` overrides with real `litellm` streaming, `MockLLMProvider` yields the canned response
> in pieces for tests. This slice is the `llm` layer ONLY — no agent/transport/security wiring (slices 2–3).

- **Date:** 2026-06-19
- **Phase:** Phase 4+ — Real Token Streaming, slice 1 of 3 (provider → sliding-window gate → transport seam).
- **Branch:** `feat/provider-streaming` (off `main`).

---

## 1. Context & decomposition

Format-level SSE streaming shipped (R13): `stream:true` runs the agent fully, then streams the completed
output. Real *token* streaming was deferred because it spans three subsystems AND collides with the
fail-closed output gate (rule 9 — no secret/invalid output ever leaves). Brainstorming resolved the
collision: a **sliding-window secret scan** with overlap ≥ the longest secret pattern is *sound* (a secret
can't span beyond the window), so real streaming and the gate are compatible. The work decomposes into
three ship-then-validate slices:

1. **Provider streaming (`llm`)** — `LLMProvider.stream()` + LiteLLMProvider + MockLLMProvider. ← THIS SLICE.
2. **Sliding-window secret scanner (`security`)** — a pure `scan_split` (safe-prefix / held-tail) + hit raise.
3. **Streaming agent/transport seam (`core`/`serve`)** — wire `provider.stream` → the windowed gate → real
   SSE in the chat handler (replacing format-level for `stream:true`); incl. the agent-seam decision (only
   passthrough chat agents can stream meaningfully). Its own brainstorm.

This slice is self-contained: a new provider method + two overrides + tests. Nothing else changes; end-to-end
`stream:true` stays on R13's format-level path until slice 3.

**Locked decisions (do not relitigate):**
- `stream()` yields **`str` content deltas** (`Iterator[str]`), not chunk objects — the simplest seam.
- **Concrete default on the ABC** (NOT abstractmethod) — a one-shot fallback over `complete()` — so no
  existing `LLMProvider` subclass breaks.
- **No per-delta usage/metrics this slice** — streamed token/cost accounting (litellm's final-chunk usage)
  is the transport slice's concern.

## 2. The interface — `LLMProvider.stream` (concrete default)

In `src/lottie/llm/base.py`, add `from collections.abc import Iterator` and a concrete method on the ABC:

```python
    def stream(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> Iterator[str]:
        """Yield assistant content deltas as they are produced.

        Default: a one-shot fallback that yields the whole `complete()` content in a single delta —
        so a provider that only implements `complete` still satisfies the streaming interface (no
        incremental latency). Real providers override this with true incremental streaming.
        """
        yield self.complete(messages, model_params).content
```

`complete` stays the only `@abstractmethod`. Every existing subclass (LiteLLM, Mock, test providers like
`_BoomProvider`) inherits a working `stream` for free; a provider whose `complete` raises will raise from
`stream` on first iteration — consistent.

## 3. `LiteLLMProvider.stream` — real streaming

In `src/lottie/llm/litellm_provider.py` (add `from collections.abc import Iterator`):

```python
    def stream(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> Iterator[str]:
        params = dict(model_params or {})
        payload = [{"role": m.role, "content": m.content} for m in messages]
        for chunk in litellm.completion(
            model=self._model, messages=payload, stream=True, **params
        ):
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
```

litellm's `stream=True` returns an iterator of chunk objects whose `choices[0].delta.content` is the
incremental text (or `None`/`""` on non-content chunks — skipped). This is the ONLY place litellm streaming
is touched (rule 1). No usage/cost here (streamed usage arrives in a final chunk only when
`stream_options={"include_usage": True}` — deferred to the transport slice).

## 4. `MockLLMProvider.stream` — chunked canned response

In `src/lottie/llm/mock.py` (add `import re` and `from collections.abc import Iterator`):

```python
    def stream(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> Iterator[str]:
        """Replay the next canned response as multiple deltas (so tests see real chunking).

        Consumes the same response queue as `complete` (advances the index, records the call), then
        yields the response split into pieces that reconstruct it EXACTLY (`"".join(deltas) == response`)."""
        self.calls.append(messages)
        if self._index >= len(self._responses):
            raise RuntimeError("MockLLMProvider responses exhausted")
        content = self._responses[self._index]
        self._index += 1
        # word-with-trailing-space and whitespace runs both captured -> join == content exactly.
        for piece in re.findall(r"\S+\s*|\s+", content):
            yield piece
```

`re.findall(r"\S+\s*|\s+", content)` tokenizes into non-space-run+trailing-space pieces (and any pure-space
runs), so `"".join(pieces) == content` for any string, and a multi-word response yields multiple deltas. An
empty response (`""`) yields nothing — acceptable (mocks are constructed with non-empty responses by
convention; `__init__` already rejects an empty `responses` list, not empty strings).

## 5. Testing

`src/lottie/llm/tests/` (mirror the existing provider tests; mock `litellm.completion`, never a real call).

- **`stream` default fallback** (unit): a minimal `LLMProvider` subclass implementing only `complete`
  (returns `LLMResponse(content="hello", ...)`) → `list(p.stream([...]))` == `["hello"]` (one delta).
- **`MockLLMProvider.stream` reconstructs** (unit): `MockLLMProvider(["the launch post"]).stream([...])` →
  `len(deltas) > 1` and `"".join(deltas) == "the launch post"`; `.calls` recorded once; the index advanced
  (a following `complete`/`stream` gets the next response, and exhaustion raises).
- **`MockLLMProvider.stream` single-token** (unit): `["hi"]` → `["hi"]` (one delta; still reconstructs).
- **`LiteLLMProvider.stream`** (unit, `litellm.completion` monkeypatched): patch it to return an iterable of
  fake chunk objects with `.choices[0].delta.content` of `["The ", "launch ", "", None, "post"]` → the
  stream yields `["The ", "launch ", "post"]` (empties/None skipped); assert `stream=True` was passed.
- **Full gate**: `uv run pytest -q`, `uv run mypy --strict src`, `uv run ruff check` — green; the existing
  ~828 tests unaffected (`complete` and all current behavior unchanged; `stream` is additive).

## 6. Definition of done

`LLMProvider.stream(messages, model_params=None) -> Iterator[str]` exists as a concrete default (one-shot
over `complete()`); `LiteLLMProvider.stream` does real `litellm.completion(stream=True)` delta streaming
(empties skipped, the only litellm-streaming touchpoint); `MockLLMProvider.stream` yields the next canned
response as reconstruct-exact pieces, sharing the response queue with `complete`. No agent/transport/security
change; `complete` and all existing behavior unchanged. `uv run pytest -q` / `mypy --strict src` / `ruff
check` green. (No lab round for this slice alone — it's internal plumbing; the real-streaming lab round
lands with slice 3.) Commit on the feature branch; do not push until approved.
```
