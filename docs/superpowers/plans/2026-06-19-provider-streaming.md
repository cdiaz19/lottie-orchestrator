# LLMProvider.stream() Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `LLMProvider.stream(messages, model_params=None) -> Iterator[str]` — a concrete one-shot default, real `litellm` streaming in `LiteLLMProvider`, chunked replay in `MockLLMProvider`.

**Architecture:** `stream` is a concrete method on the `LLMProvider` ABC (default yields the whole `complete()` content once, so no existing provider breaks); `LiteLLMProvider` overrides with `litellm.completion(stream=True)` delta streaming; `MockLLMProvider` overrides to replay the next canned response as reconstruct-exact pieces (sharing the response queue with `complete`).

**Tech Stack:** Python 3.12, litellm (adapter-only), pytest, `uv run` (mypy --strict, ruff).

**Design:** `docs/superpowers/specs/2026-06-19-provider-streaming-design.md`

This is **slice 1 of 3** for real token streaming (provider → sliding-window gate → transport seam). `llm` layer ONLY — no agent/transport/security change. No lab round (internal plumbing; the real-streaming round lands with slice 3).

---

## File structure

- **Modify** `src/lottie/llm/base.py` — add `Iterator` import + the concrete `stream` default on the ABC.
- **Modify** `src/lottie/llm/litellm_provider.py` — `stream` override (real litellm streaming).
- **Modify** `src/lottie/llm/mock.py` — `stream` override (chunked canned replay).
- **Test:** `src/lottie/llm/tests/test_llm_base.py`, `test_litellm_provider.py`, `test_mock_provider.py`.

Known facts (verified):
- `base.py`: `from collections.abc import Mapping`; `Message(role, content)`; `LLMResponse(content, usage, model, cost_usd)`; `class LLMProvider(ABC)` with `@abstractmethod complete(self, messages, model_params=None) -> LLMResponse` and an abstract `model` property.
- `litellm_provider.py`: `complete` builds `payload = [{"role": m.role, "content": m.content} for m in messages]`, calls `litellm.completion(model=self._model, messages=payload, **params)`; litellm imported at module top.
- `mock.py`: `MockLLMProvider(responses: list[str], model="mock/mock-model")`; `__init__` rejects empty `responses` list; `complete` appends to `self.calls`, raises `RuntimeError("MockLLMProvider responses exhausted")` when `self._index >= len(self._responses)`, else returns `self._responses[self._index]` and increments `self._index`.
- `test_litellm_provider.py` mocks litellm via `monkeypatch.setattr("lottie.llm.litellm_provider.litellm.completion", fake)` and builds fake responses with `types.SimpleNamespace`. `test_llm_base.py` defines minimal `LLMProvider` subclasses inline (e.g. `EchoProvider`) and imports `Mapping`, `Message`, `LLMResponse`, `TokenUsage`, `LLMProvider`.

---

## Task 1: ABC concrete `stream` default

**Files:**
- Modify: `src/lottie/llm/base.py`
- Test: `src/lottie/llm/tests/test_llm_base.py`

- [ ] **Step 1: Write the failing test** — append to `src/lottie/llm/tests/test_llm_base.py`:

```python
def test_stream_default_yields_complete_content_once() -> None:
    class OneShot(LLMProvider):
        @property
        def model(self) -> str:
            return "one/shot"

        def complete(
            self,
            messages: list[Message],
            model_params: Mapping[str, object] | None = None,
        ) -> LLMResponse:
            return LLMResponse(content="hello world", model=self.model)

    deltas = list(OneShot().stream([Message(role="user", content="hi")]))
    assert deltas == ["hello world"]  # default = one delta over complete()
```

(`LLMProvider`, `Message`, `LLMResponse`, `Mapping` are already imported in this test file.)

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest src/lottie/llm/tests/test_llm_base.py -q -k stream_default` (AttributeError: no `stream`).

- [ ] **Step 3: Add the concrete default** — in `src/lottie/llm/base.py`, change the import line `from collections.abc import Mapping` to `from collections.abc import Iterator, Mapping`, and add a concrete method to `LLMProvider` (after `complete`):

```python
    def stream(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> Iterator[str]:
        """Yield assistant content deltas as they are produced.

        Default: a one-shot fallback that yields the whole `complete()` content in a single delta,
        so a provider implementing only `complete` still satisfies the streaming interface (no
        incremental latency). Real providers override this with true incremental streaming.
        """
        yield self.complete(messages, model_params).content
```

(Keep `complete` the only `@abstractmethod`. `stream` is NOT decorated abstract.)

- [ ] **Step 4: Run, verify PASS** — `uv run pytest src/lottie/llm/tests/test_llm_base.py -q`.

- [ ] **Step 5: Gates** — `uv run mypy --strict src/lottie/llm` and `uv run ruff check src/lottie/llm` clean.

- [ ] **Step 6: Commit**

```bash
git add src/lottie/llm/base.py src/lottie/llm/tests/test_llm_base.py
git commit -m "feat(llm): LLMProvider.stream() concrete default (one-shot over complete)"
```

---

## Task 2: `MockLLMProvider.stream` — chunked replay

**Files:**
- Modify: `src/lottie/llm/mock.py`
- Test: `src/lottie/llm/tests/test_mock_provider.py`

- [ ] **Step 1: Write the failing tests** — append to `src/lottie/llm/tests/test_mock_provider.py` (add `import pytest` and `from lottie.llm import Message` to the top if absent):

```python
def test_mock_stream_reconstructs_response() -> None:
    p = MockLLMProvider(["the launch post"])
    deltas = list(p.stream([Message(role="user", content="x")]))
    assert len(deltas) > 1  # multiple deltas for a multi-word response
    assert "".join(deltas) == "the launch post"  # reconstructs exactly


def test_mock_stream_single_token_one_delta() -> None:
    p = MockLLMProvider(["hi"])
    assert list(p.stream([Message(role="user", content="x")])) == ["hi"]


def test_mock_stream_advances_queue_and_records_call() -> None:
    p = MockLLMProvider(["first", "second"])
    list(p.stream([Message(role="user", content="a")]))
    assert len(p.calls) == 1
    # the queue advanced — a following complete() gets the SECOND response
    assert p.complete([Message(role="user", content="b")]).content == "second"


def test_mock_stream_raises_when_exhausted() -> None:
    p = MockLLMProvider(["only"])
    list(p.stream([Message(role="user", content="a")]))
    with pytest.raises(RuntimeError):
        list(p.stream([Message(role="user", content="b")]))
```

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest src/lottie/llm/tests/test_mock_provider.py -q -k stream` (AttributeError: no `stream`).

- [ ] **Step 3: Add the override** — in `src/lottie/llm/mock.py`, add `import re` and change the import to `from collections.abc import Iterator, Mapping`. Add to `MockLLMProvider`:

```python
    def stream(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> Iterator[str]:
        """Replay the next canned response as multiple deltas (so tests see real chunking).

        Consumes the same response queue as `complete` (records the call, advances the index), then
        yields pieces that reconstruct the response EXACTLY (`"".join(deltas) == response`)."""
        self.calls.append(messages)
        if self._index >= len(self._responses):
            raise RuntimeError("MockLLMProvider responses exhausted")
        content = self._responses[self._index]
        self._index += 1
        # non-space-run+trailing-space and pure-space runs both captured -> join == content.
        for piece in re.findall(r"\S+\s*|\s+", content):
            yield piece
```

- [ ] **Step 4: Run, verify PASS** — `uv run pytest src/lottie/llm/tests/test_mock_provider.py -q`.

- [ ] **Step 5: Gates** — `uv run mypy --strict src/lottie/llm` and `uv run ruff check src/lottie/llm` clean.

- [ ] **Step 6: Commit**

```bash
git add src/lottie/llm/mock.py src/lottie/llm/tests/test_mock_provider.py
git commit -m "feat(llm): MockLLMProvider.stream replays canned response in deltas"
```

---

## Task 3: `LiteLLMProvider.stream` — real litellm streaming

**Files:**
- Modify: `src/lottie/llm/litellm_provider.py`
- Test: `src/lottie/llm/tests/test_litellm_provider.py`

- [ ] **Step 1: Write the failing test** — append to `src/lottie/llm/tests/test_litellm_provider.py` (`SimpleNamespace`, `Any`, `pytest`, `LiteLLMProvider`, `Message` are already imported):

```python
def test_stream_yields_content_deltas_skipping_empties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _chunk(text: Any) -> Any:
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])

    def fake_completion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return iter([_chunk("The "), _chunk("launch "), _chunk(""), _chunk(None), _chunk("post")])

    monkeypatch.setattr("lottie.llm.litellm_provider.litellm.completion", fake_completion)

    provider = LiteLLMProvider(model="openai/gpt-4o")
    deltas = list(provider.stream([Message(role="user", content="q")]))

    assert deltas == ["The ", "launch ", "post"]  # empties + None skipped
    assert captured["stream"] is True
    assert captured["model"] == "openai/gpt-4o"
    assert captured["messages"] == [{"role": "user", "content": "q"}]
```

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest src/lottie/llm/tests/test_litellm_provider.py -q -k stream` (AttributeError: no `stream`).

- [ ] **Step 3: Add the override** — in `src/lottie/llm/litellm_provider.py`, change the import to `from collections.abc import Iterator, Mapping`. Add to `LiteLLMProvider`:

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

(This is the only place litellm streaming is touched — rule 1. No usage/cost on the stream path; streamed usage is the transport slice's concern.)

- [ ] **Step 4: Run, verify PASS** — `uv run pytest src/lottie/llm/tests/test_litellm_provider.py -q`.

- [ ] **Step 5: Gates** — `uv run mypy --strict src/lottie/llm` and `uv run ruff check src/lottie/llm` clean.

- [ ] **Step 6: Commit**

```bash
git add src/lottie/llm/litellm_provider.py src/lottie/llm/tests/test_litellm_provider.py
git commit -m "feat(llm): LiteLLMProvider.stream real token streaming via litellm stream=True"
```

---

## Task 4: Closeout — full gate

**Files:** none (verification only)

- [ ] **Step 1: Full test suite** — `uv run pytest -q`. Expected: PASS — all prior ~828 tests plus the new provider-stream tests; `complete` and all existing behavior unchanged (`stream` is purely additive).

- [ ] **Step 2: Types** — `uv run mypy --strict src`. Expected: clean (`stream` returns `Iterator[str]`; the litellm chunk is untyped `Any`, like the existing `complete` path — consistent).

- [ ] **Step 3: Lint** — `uv run ruff check`. Expected: clean.

- [ ] **Step 4: Final commit (if any closeout fixes)**

```bash
git add -A
git commit -m "chore(llm): closeout fixes for LLMProvider.stream (mypy/ruff)"
```

---

## Notes for the implementer

- **Additive only.** `complete` and every existing provider behavior is unchanged; `stream` is a new method. Existing tests must stay green untouched.
- **`stream` is a concrete default, NOT abstract** — so test providers (and any subclass) that implement only `complete` keep working via the one-shot fallback.
- **litellm streaming is adapter-only** (rule 1) — `LiteLLMProvider.stream` is the sole streaming touchpoint; agent/skill code never sees litellm.
- **No agent/transport/security wiring** — that's slices 2 (sliding-window gate) and 3 (the SSE seam). Do NOT touch `serve`/`core`/`security` here.
```
