# Real Token Streaming — Slice 3a Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add `BaseAgent.run_stream` — a streaming analog of `run()` that flows LLM content deltas through the
governance chokepoint (policy + cost + audit + usage), plus an opt-in `_stream` agent method and a
usage-bearing `LLMProvider.stream_complete` primitive. Core + `llm` layers only; nothing client-visible (3b).

**Architecture:** A generator pipeline. `LLMProvider.stream_complete` yields `str` deltas and `return`s a
`StreamResult(usage, cost)` via `StopIteration.value`. `InstrumentedRunnable._instrument_stream` reproduces
`run()`'s timing/metrics/span as a generator. `BaseAgent.run_stream` wraps it with the shared `_pre_run_gates`
(policy+cost) and a post-stream `_write_audit`. `supports_streaming()` is an explicit override check.

**Tech Stack:** Python 3.12, Pydantic v2, litellm (adapter-only, rule 1), pytest, `uv run` (mypy --strict, ruff).

**Spec:** `docs/superpowers/specs/2026-06-20-real-token-streaming-3a-design.md`

---

## File Structure

- `src/lottie/llm/base.py` — `StreamResult` model + `LLMProvider.stream_complete` concrete default (Task 1).
- `src/lottie/llm/mock.py` — `MockLLMProvider.stream_complete` override (Task 2).
- `src/lottie/llm/litellm_provider.py` — `LiteLLMProvider.stream_complete` override (Task 3).
- `src/lottie/core/runnable.py` — `InstrumentedRunnable._instrument_stream` (Task 4).
- `src/lottie/core/base_agent.py` — `NotStreamable`, `_stream`, `supports_streaming`, `stream_complete`
  (Task 5); `_pre_run_gates` extract (Task 6); `run_stream` (Task 7).
- Tests: `llm/tests/test_llm_base.py`, `test_mock_provider.py`, `test_litellm_provider.py` (extend);
  `core/tests/test_runnable_stream.py`, `core/tests/test_base_agent_stream.py` (new).

**Test helper (used in Tasks 1, 5):** drain a generator capturing its return value:

```python
def _drain(gen: object) -> tuple[list[str], object]:
    out: list[str] = []
    try:
        while True:
            out.append(next(gen))  # type: ignore[arg-type]
    except StopIteration as stop:
        return out, stop.value
```

---

### Task 1: `StreamResult` + provider default `stream_complete`

**Files:**
- Modify: `src/lottie/llm/base.py`
- Test: `src/lottie/llm/tests/test_llm_base.py`

- [ ] **Step 1: Write the failing test**

Add to `src/lottie/llm/tests/test_llm_base.py` (define `_drain` at module top if not present):

```python
from lottie.llm.base import LLMProvider, LLMResponse, Message, StreamResult, TokenUsage


class _CompleteOnly(LLMProvider):
    @property
    def model(self) -> str:
        return "stub/model"

    def complete(self, messages, model_params=None):
        return LLMResponse(
            content="hello world", usage=TokenUsage(input_tokens=4, output_tokens=2),
            model="stub/model", cost_usd=0.3,
        )


def test_stream_complete_default_one_shot_yields_content_and_returns_usage() -> None:
    deltas, result = _drain(_CompleteOnly().stream_complete([Message(role="user", content="x")]))
    assert deltas == ["hello world"]            # one delta (no incremental latency)
    assert isinstance(result, StreamResult)
    assert result.usage.input_tokens == 4 and result.usage.output_tokens == 2
    assert result.cost_usd == 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/llm/tests/test_llm_base.py::test_stream_complete_default_one_shot_yields_content_and_returns_usage -v`
Expected: FAIL — `ImportError: cannot import name 'StreamResult'`.

- [ ] **Step 3: Write minimal implementation**

In `src/lottie/llm/base.py`, change the import line `from collections.abc import Iterator, Mapping` to
`from collections.abc import Generator, Iterator, Mapping`. Add the `StreamResult` model after `LLMResponse`:

```python
class StreamResult(BaseModel):
    """Usage/cost for a streamed completion, delivered at stream end (the generator's return value)."""

    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
```

Add the concrete default method on `LLMProvider` (after `stream`):

```python
    def stream_complete(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> Generator[str, None, StreamResult]:
        """Yield assistant content deltas, returning usage/cost at stream end.

        Default: one-shot over `complete()` — yields the whole content in a single delta and returns its
        usage, so a `complete`-only provider satisfies the governed streaming interface. Real providers
        override with incremental streaming + final-chunk usage. Distinct from `stream` (content-only):
        `stream_complete` carries usage so the agent can keep audit/cost parity with non-streaming runs.
        """
        response = self.complete(messages, model_params)
        yield response.content
        return StreamResult(usage=response.usage, cost_usd=response.cost_usd)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/llm/tests/test_llm_base.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lottie/llm/base.py src/lottie/llm/tests/test_llm_base.py
git commit -m "feat(llm): StreamResult + LLMProvider.stream_complete default (usage-bearing stream primitive)"
```

---

### Task 2: `MockLLMProvider.stream_complete`

**Files:**
- Modify: `src/lottie/llm/mock.py`
- Test: `src/lottie/llm/tests/test_mock_provider.py`

- [ ] **Step 1: Write the failing test**

Add to `src/lottie/llm/tests/test_mock_provider.py` (add `_drain` from the File Structure section to the top
if not already there; import `StreamResult`):

```python
from lottie.llm.base import StreamResult


def test_mock_stream_complete_reconstructs_and_returns_zero_usage() -> None:
    p = MockLLMProvider(["the launch post"])
    deltas, result = _drain(p.stream_complete([Message(role="user", content="x")]))
    assert "".join(deltas) == "the launch post" and len(deltas) > 1
    assert isinstance(result, StreamResult)
    assert result.usage.input_tokens == 0 and result.usage.output_tokens == 0 and result.cost_usd == 0.0


def test_mock_stream_complete_shares_queue_with_complete() -> None:
    p = MockLLMProvider(["first", "second"])
    _drain(p.stream_complete([Message(role="user", content="a")]))     # consumes "first"
    assert p.complete([Message(role="user", content="b")]).content == "second"
    assert len(p.calls) == 2
    with pytest.raises(RuntimeError):                                   # exhausted
        _drain(p.stream_complete([Message(role="user", content="c")]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/llm/tests/test_mock_provider.py::test_mock_stream_complete_reconstructs_and_returns_zero_usage -v`
Expected: FAIL — `AttributeError`/`StopIteration` value is `None` (default returns `StreamResult` only after the
override exists; before the override the inherited default yields the whole content as ONE delta, so
`len(deltas) > 1` fails).

- [ ] **Step 3: Write minimal implementation**

In `src/lottie/llm/mock.py`: change `from collections.abc import Iterator, Mapping` to
`from collections.abc import Generator, Iterator, Mapping`; add `StreamResult` to the base import
(`from lottie.llm.base import LLMProvider, LLMResponse, Message, StreamResult, TokenUsage`). Add the method
after `stream`:

```python
    def stream_complete(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> Generator[str, None, StreamResult]:
        """Replay the next canned response as deltas (queue shared with `complete`); zero usage.

        MockLLM has no real usage; a usage-parity test uses a fixture provider with non-zero usage instead."""
        content = self._pop_response(messages)
        for piece in re.findall(r"\S+\s*|\s+", content):
            yield piece
        return StreamResult()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/llm/tests/test_mock_provider.py -v`
Expected: PASS (all, including the existing `stream` tests).

- [ ] **Step 5: Commit**

```bash
git add src/lottie/llm/mock.py src/lottie/llm/tests/test_mock_provider.py
git commit -m "feat(llm): MockLLMProvider.stream_complete — chunked replay + zero-usage StreamResult"
```

---

### Task 3: `LiteLLMProvider.stream_complete`

**Files:**
- Modify: `src/lottie/llm/litellm_provider.py`
- Test: `src/lottie/llm/tests/test_litellm_provider.py`

- [ ] **Step 1: Write the failing test**

Add to `src/lottie/llm/tests/test_litellm_provider.py` (it already imports `SimpleNamespace`, `pytest`,
`Message`, `LiteLLMProvider`; add `_drain` from the File Structure section + `from lottie.llm.base import
StreamResult`):

```python
def _delta_chunk(text: str | None) -> SimpleNamespace:
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


def _usage_chunk(prompt: int, completion: int) -> SimpleNamespace:
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion), choices=[]
    )


def test_stream_complete_yields_deltas_and_returns_final_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_completion(*, model, messages, **kwargs):
        captured.update(kwargs)
        return iter([_delta_chunk("The "), _delta_chunk(""), _delta_chunk(None),
                     _delta_chunk("post"), _usage_chunk(11, 5)])

    monkeypatch.setattr("lottie.llm.litellm_provider.litellm.completion", fake_completion)
    monkeypatch.setattr("lottie.llm.litellm_provider.litellm.completion_cost", lambda **_: 0.4)
    provider = LiteLLMProvider("anthropic/claude-sonnet-4-6")

    deltas, result = _drain(provider.stream_complete([Message(role="user", content="q")]))
    assert deltas == ["The ", "post"]                       # empties + None + the choice-less usage chunk skipped
    assert isinstance(result, StreamResult)
    assert result.usage.input_tokens == 11 and result.usage.output_tokens == 5
    assert result.cost_usd == 0.4
    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}


def test_stream_complete_forces_stream_flags_over_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_completion(*, model, messages, **kwargs):
        captured.update(kwargs)
        return iter([_delta_chunk("ok")])

    monkeypatch.setattr("lottie.llm.litellm_provider.litellm.completion", fake_completion)
    monkeypatch.setattr("lottie.llm.litellm_provider.litellm.completion_cost", lambda **_: 0.0)
    provider = LiteLLMProvider("m")
    deltas, _ = _drain(provider.stream_complete(
        [Message(role="user", content="q")], model_params={"stream": False, "stream_options": {}},
    ))
    assert deltas == ["ok"]                                  # no "multiple values for stream" TypeError
    assert captured["stream"] is True and captured["stream_options"] == {"include_usage": True}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/llm/tests/test_litellm_provider.py::test_stream_complete_yields_deltas_and_returns_final_usage -v`
Expected: FAIL — `AttributeError: 'LiteLLMProvider' object has no attribute 'stream_complete'` is not raised
(inherited default exists), but the default one-shot calls `litellm.completion` WITHOUT `stream` → the fake
asserts fail / `captured["stream"]` KeyError.

- [ ] **Step 3: Write minimal implementation**

In `src/lottie/llm/litellm_provider.py`: change `from collections.abc import Iterator, Mapping` to
`from collections.abc import Generator, Iterator, Mapping`; add `StreamResult` to the base import. Add the
method after `stream`:

```python
    def stream_complete(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> Generator[str, None, StreamResult]:
        params = dict(model_params or {})
        params.pop("stream", None)          # the streaming path owns these flags; drop caller-set ones
        params.pop("stream_options", None)
        payload = [{"role": m.role, "content": m.content} for m in messages]
        usage = TokenUsage()
        cost = 0.0
        for chunk in litellm.completion(
            model=self._model, messages=payload, stream=True,
            stream_options={"include_usage": True}, **params,
        ):
            chunk_usage = getattr(chunk, "usage", None)     # usage rides a final, choice-less chunk
            if chunk_usage is not None:
                usage = TokenUsage(
                    input_tokens=chunk_usage.prompt_tokens or 0,
                    output_tokens=chunk_usage.completion_tokens or 0,
                )
                cost = self._cost(chunk) or cost            # same cost path as complete()
            if not chunk.choices:                           # usage-only chunk -> no delta
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
        return StreamResult(usage=usage, cost_usd=cost)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/llm/tests/test_litellm_provider.py -v`
Expected: PASS (all, including the existing `complete`/`stream` tests).

- [ ] **Step 5: Commit**

```bash
git add src/lottie/llm/litellm_provider.py src/lottie/llm/tests/test_litellm_provider.py
git commit -m "feat(llm): LiteLLMProvider.stream_complete — real streaming with final-chunk usage"
```

---

### Task 4: `InstrumentedRunnable._instrument_stream`

**Files:**
- Modify: `src/lottie/core/runnable.py`
- Test: `src/lottie/core/tests/test_runnable_stream.py` (create)

- [ ] **Step 1: Write the failing test**

Create `src/lottie/core/tests/test_runnable_stream.py`:

```python
from __future__ import annotations

from collections.abc import Iterator

from pydantic import BaseModel

from lottie.core.runnable import InstrumentedRunnable
from lottie.llm.base import TokenUsage


class _In(BaseModel):
    q: str


class _Out(BaseModel):
    a: str


class _Stub(InstrumentedRunnable[_In, _Out]):
    kind = "agent"

    def __init__(self) -> None:
        super().__init__(enable_benchmarks=False)

    def _execute(self, data: _In) -> _Out:  # unused by streaming tests
        return _Out(a=data.q)


def _producer(stub: _Stub) -> Iterator[str]:
    stub._active_ctx.add_usage(TokenUsage(input_tokens=3, output_tokens=2), 0.5)  # type: ignore[union-attr]
    yield "a"
    yield "b"


def test_instrument_stream_records_metrics_and_usage() -> None:
    stub = _Stub()
    assert list(stub._instrument_stream(_producer(stub))) == ["a", "b"]
    m = stub.last_metrics
    assert m is not None and m.success is True
    assert m.input_tokens == 3 and m.output_tokens == 2 and m.cost_usd == 0.5
    assert stub._active_ctx is None  # cleared after the stream


def test_instrument_stream_early_close_records_partial() -> None:
    stub = _Stub()
    gen = stub._instrument_stream(_producer(stub))
    assert next(gen) == "a"
    gen.close()
    m = stub.last_metrics
    assert m is not None and m.success is False
    assert m.error == "stream closed before completion"
    assert m.input_tokens == 3  # usage accumulated before the close is retained
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/core/tests/test_runnable_stream.py -v`
Expected: FAIL — `AttributeError: '_Stub' object has no attribute '_instrument_stream'`.

- [ ] **Step 3: Write minimal implementation**

In `src/lottie/core/runnable.py`: add `from collections.abc import Iterator` near the top imports. Add the
method after `run`:

```python
    def _instrument_stream(self, pieces: Iterator[str]) -> Iterator[str]:
        """run()'s instrumentation, streamed: time the run, record metrics post.

        `pieces` is the producer (an agent's `_stream`); it accumulates usage into `_active_ctx` as it
        runs. On early consumer close (`GeneratorExit`) the run is recorded PARTIAL, not a clean success —
        `_record` only writes (never yields), so it is safe during close (unlike a flushing generator).
        """
        ctx = RunContext()
        self._active_ctx = ctx
        start = perf_counter()
        success = True
        error: str | None = None
        with run_span(self.name, self.kind) as span:
            try:
                yield from pieces
            except GeneratorExit:  # BaseException, not Exception — handled explicitly, then re-raised
                success = False
                error = "stream closed before completion"
                raise
            except Exception as exc:
                success = False
                error = repr(exc)
                span_set_error(span, exc)
                raise
            finally:
                self._record(ctx, start, success, error)
                span_set_metrics(span, self.last_metrics)
                self._active_ctx = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/core/tests/test_runnable_stream.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lottie/core/runnable.py src/lottie/core/tests/test_runnable_stream.py
git commit -m "feat(core): InstrumentedRunnable._instrument_stream — run() instrumentation as a generator"
```

---

### Task 5: `BaseAgent` streaming API — `NotStreamable`, `_stream`, `supports_streaming`, `stream_complete`

**Files:**
- Modify: `src/lottie/core/base_agent.py`
- Test: `src/lottie/core/tests/test_base_agent_stream.py` (create)

- [ ] **Step 1: Write the failing test**

Create `src/lottie/core/tests/test_base_agent_stream.py`:

```python
from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import BaseModel

from lottie.core.base_agent import BaseAgent, NotStreamable
from lottie.core.metrics import RunContext
from lottie.governance.audit import SqliteAuditLogger
from lottie.llm import MockLLMProvider
from lottie.llm.base import LLMProvider, LLMResponse, Message, StreamResult, TokenUsage


class _In(BaseModel):
    q: str


class _Out(BaseModel):
    a: str


class _Plain(BaseAgent[_In, _Out]):
    def _execute(self, data: _In) -> _Out:
        return _Out(a=data.q)


class _StreamingAgent(BaseAgent[_In, _Out]):
    def _execute(self, data: _In) -> _Out:
        return _Out(a=self.complete([Message(role="user", content=data.q)]).content)

    def _stream(self, data: _In) -> Iterator[str]:
        yield from self.stream_complete([Message(role="user", content=data.q)])


class _UsageProvider(LLMProvider):
    @property
    def model(self) -> str:
        return "fix/model"

    def complete(self, messages, model_params=None):
        return LLMResponse(content="the launch post",
                           usage=TokenUsage(input_tokens=11, output_tokens=7),
                           model="fix/model", cost_usd=0.25)

    def stream_complete(self, messages, model_params=None):
        import re
        for piece in re.findall(r"\S+\s*|\s+", "the launch post"):
            yield piece
        return StreamResult(usage=TokenUsage(input_tokens=11, output_tokens=7), cost_usd=0.25)


def test_supports_streaming_reflects_override() -> None:
    assert _StreamingAgent.supports_streaming() is True
    assert _Plain.supports_streaming() is False


def test_default_stream_raises_not_streamable(tmp_path) -> None:
    agent = _Plain(MockLLMProvider(["x"]), audit=SqliteAuditLogger(tmp_path))
    with pytest.raises(NotStreamable):
        agent._stream(_In(q="hi"))


def test_stream_complete_accumulates_usage_into_active_ctx(tmp_path) -> None:
    agent = _StreamingAgent(_UsageProvider(), audit=SqliteAuditLogger(tmp_path))
    agent._active_ctx = RunContext()
    deltas = list(agent.stream_complete([Message(role="user", content="hi")]))
    assert "".join(deltas) == "the launch post"
    assert agent._active_ctx.input_tokens == 11
    assert agent._active_ctx.output_tokens == 7
    assert agent._active_ctx.cost_usd == 0.25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/core/tests/test_base_agent_stream.py -v`
Expected: FAIL — `ImportError: cannot import name 'NotStreamable'`.

- [ ] **Step 3: Write minimal implementation**

In `src/lottie/core/base_agent.py`: change `from collections.abc import Mapping` to
`from collections.abc import Iterator, Mapping`. Add the exception class after the imports (module level,
before `BaseAgent`):

```python
class NotStreamable(RuntimeError):
    """Raised if `_stream` runs on an agent that did not opt in (the capability check is the guard)."""
```

Add these methods on `BaseAgent` (place near `complete`):

```python
    def _stream(self, data: InputT) -> Iterator[str]:
        """Opt-in streaming producer. Default raises; override to enable real token streaming."""
        raise NotStreamable(f"{self.name} does not implement _stream")

    @classmethod
    def supports_streaming(cls) -> bool:
        """True if this agent overrides `_stream` (drives real-stream vs format-fallback in the transport)."""
        return cls._stream is not BaseAgent._stream

    def stream_complete(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> Iterator[str]:
        """Agent analog of `complete`: stream deltas, accumulating usage into the active run at stream end."""
        result = yield from self.llm.stream_complete(messages, model_params)
        if self._active_ctx is not None:
            self._active_ctx.add_usage(result.usage, result.cost_usd)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/core/tests/test_base_agent_stream.py -v`
Expected: PASS (the three tests above; `run_stream` tests come in Task 7).

- [ ] **Step 5: Commit**

```bash
git add src/lottie/core/base_agent.py src/lottie/core/tests/test_base_agent_stream.py
git commit -m "feat(core): BaseAgent streaming API — _stream opt-in, supports_streaming, stream_complete"
```

---

### Task 6: Extract `_pre_run_gates` (refactor `run`, shared with `run_stream`)

**Files:**
- Modify: `src/lottie/core/base_agent.py`
- Test: existing `test_base_agent_policy.py` / `test_base_agent_cost.py` (behavior unchanged)

- [ ] **Step 1: Refactor — extract the pre-check**

In `src/lottie/core/base_agent.py`, add this method just above `run`:

```python
    def _pre_run_gates(self, data: InputT) -> None:
        """Policy then budget pre-checks; audit a block and re-raise if either trips (shared by run/run_stream)."""
        try:
            self._policy.check()   # capability policy — checked FIRST (no I/O)
            self._cost.check()     # cumulative budget — checked SECOND (reads the ledger)
        except PolicyViolation as exc:
            self._write_block(
                data, exc, "escalated" if isinstance(exc, PolicyEscalation) else "denied"
            )
            raise
        except BudgetExceeded as exc:
            self._write_block(data, exc, "budget_exceeded")
            raise
```

Replace the body of `run` so its pre-check delegates (everything after the pre-check is unchanged):

```python
    def run(self, data: InputT) -> OutputT:
        """Policy + budget pre-checks, then instrumented run + audit (best-effort)."""
        self._pre_run_gates(data)
        token = _audit_depth.set(_depth() + 1)
        is_root = _depth() == 1
        output: OutputT | None = None
        try:
            output = super().run(data)
            return output
        finally:
            try:
                self._write_audit(data, output, is_root)
            finally:
                _audit_depth.reset(token)
```

- [ ] **Step 2: Run the governance suites to verify no behavior change**

Run: `uv run pytest src/lottie/core/tests/test_base_agent_policy.py src/lottie/core/tests/test_base_agent_cost.py src/lottie/core/tests/test_base_agent_audit.py -v`
Expected: PASS (all existing — block statuses, ordering, audit rows unchanged).

- [ ] **Step 3: Commit**

```bash
git add src/lottie/core/base_agent.py
git commit -m "refactor(core): extract BaseAgent._pre_run_gates (shared by run and run_stream)"
```

---

### Task 7: `BaseAgent.run_stream`

**Files:**
- Modify: `src/lottie/core/base_agent.py`
- Test: `src/lottie/core/tests/test_base_agent_stream.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `src/lottie/core/tests/test_base_agent_stream.py` (imports already present from Task 5; add
`from lottie.governance.cost import BudgetExceeded, CostGate`, `from lottie.governance.policy import
PolicyDenied, PolicyGate`, `from lottie.governance.schema import AuditRecord`):

```python
def test_run_stream_yields_incrementally(tmp_path) -> None:
    agent = _StreamingAgent(MockLLMProvider(["the launch post"]), audit=SqliteAuditLogger(tmp_path))
    pieces = list(agent.run_stream(_In(q="hi")))
    assert len(pieces) > 1 and "".join(pieces) == "the launch post"
    rows = SqliteAuditLogger(tmp_path).query()
    assert rows[0].status == "ok" and rows[0].root is True


def test_run_stream_policy_deny_blocks_before_any_piece(tmp_path) -> None:
    agent = _StreamingAgent(MockLLMProvider(["x y z"]), audit=SqliteAuditLogger(tmp_path))
    agent.set_policy(PolicyGate(["shell"], allow=set(), deny={"shell"}, escalate=set()))
    gen = agent.run_stream(_In(q="hi"))
    with pytest.raises(PolicyDenied):
        next(gen)
    rows = SqliteAuditLogger(tmp_path).query()
    assert [r.status for r in rows] == ["denied"]


def test_run_stream_over_budget_blocks_pre_run(tmp_path) -> None:
    logger = SqliteAuditLogger(tmp_path)
    agent = _StreamingAgent(MockLLMProvider(["x y z"]), name="digest", audit=logger)
    logger.log(AuditRecord(
        ts="2026-06-20T10:00:00+00:00", agent="digest", provider="fix/model", status="ok",
        root=True, input_sha256="a" * 64, output_sha256="b" * 64, input_tokens=0,
        output_tokens=0, cost_usd=0.10, latency_ms=1.0, error=None,
    ))
    agent.set_cost_gate(CostGate("digest", 0.10, logger))
    gen = agent.run_stream(_In(q="hi"))
    with pytest.raises(BudgetExceeded):
        next(gen)
    assert "budget_exceeded" in [r.status for r in SqliteAuditLogger(tmp_path).query(limit=20)]


def test_run_stream_early_close_audits_partial(tmp_path) -> None:
    agent = _StreamingAgent(MockLLMProvider(["alpha beta gamma"]), audit=SqliteAuditLogger(tmp_path))
    gen = agent.run_stream(_In(q="hi"))
    assert next(gen)                       # pull the first delta
    gen.close()
    rows = SqliteAuditLogger(tmp_path).query()
    assert rows[0].status == "error" and rows[0].error == "stream closed before completion"


def test_run_stream_usage_parity_with_run(tmp_path) -> None:
    a_run = _StreamingAgent(_UsageProvider(), audit=SqliteAuditLogger(tmp_path / "r"))
    a_run.run(_In(q="hi"))
    a_stream = _StreamingAgent(_UsageProvider(), audit=SqliteAuditLogger(tmp_path / "s"))
    list(a_stream.run_stream(_In(q="hi")))
    rm, sm = a_run.last_metrics, a_stream.last_metrics
    assert rm is not None and sm is not None
    assert (rm.input_tokens, rm.output_tokens, rm.cost_usd) == (11, 7, 0.25)
    assert (sm.input_tokens, sm.output_tokens, sm.cost_usd) == (11, 7, 0.25)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/lottie/core/tests/test_base_agent_stream.py::test_run_stream_yields_incrementally -v`
Expected: FAIL — `AttributeError: '_StreamingAgent' object has no attribute 'run_stream'`.

- [ ] **Step 3: Write minimal implementation**

In `src/lottie/core/base_agent.py`, add after `run`:

```python
    def run_stream(self, data: InputT) -> Iterator[str]:
        """Streaming analog of run(): same policy/cost pre-gates, instrumented stream, audit post.

        A generator — the pre-gates run on the first `next()`, before any delta, so a deny/over-budget
        raises before the first piece. The output security gate is NOT here; it wraps the deltas at the
        serve boundary (slice 3b), exactly like the non-streaming output gate.
        """
        self._pre_run_gates(data)
        token = _audit_depth.set(_depth() + 1)
        is_root = _depth() == 1
        try:
            yield from self._instrument_stream(self._stream(data))
        finally:
            try:
                self._write_audit(data, None, is_root)  # output=None: a stream has no single typed Output
            finally:
                _audit_depth.reset(token)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/lottie/core/tests/test_base_agent_stream.py -v`
Expected: PASS (all eight).

- [ ] **Step 5: Commit**

```bash
git add src/lottie/core/base_agent.py src/lottie/core/tests/test_base_agent_stream.py
git commit -m "feat(core): BaseAgent.run_stream — governed token streaming through the chokepoint"
```

---

### Task 8: Closeout gate

**Files:** none (verification only)

- [ ] **Step 1: Full test suite**

Run: `uv run pytest -q`
Expected: PASS — all existing + new tests green; no regressions.

- [ ] **Step 2: Type check**

Run: `uv run mypy --strict src`
Expected: no errors. (Watch `Generator[str, None, StreamResult]` + `yield from` return-capture typing in
`stream_complete`; `_stream is not BaseAgent._stream` identity compare.)

- [ ] **Step 3: Lint**

Run: `uv run ruff check`
Expected: clean.

- [ ] **Step 4: Final whole-branch review**

Dispatch the final opus whole-branch reviewer (per subagent-driven-development) over the full diff vs `main`.
Focus areas: governance parity (policy/cost/audit all fire on the streaming path), the `GeneratorExit`
partial-record path, usage accumulation timing (lazy `_active_ctx`), litellm streaming touchpoint isolation
(rule 1), and that nothing in `serve/`/clients changed (3a is behind the transport).

---

## Self-Review

**Spec coverage:** §3 StreamResult/generator-return → T1. §4 provider `stream_complete` (default/litellm/mock)
→ T1/T3/T2. §5 `_instrument_stream` + GeneratorExit partial → T4. §6 `run_stream`/`_stream`/`supports_streaming`/
`stream_complete`/`_pre_run_gates` → T5/T6/T7. §7 governance layering (no output gate in core) → T7 impl +
docstring. §9 every listed test → T1–T7. §10 files → all tasks. §11 out-of-scope (SSE/transport/OutputValidation)
→ untouched. All covered.

**Placeholder scan:** none — every step has real code, real commands, real expected output.

**Type consistency:** `stream_complete` returns `Generator[str, None, StreamResult]` at the provider layer
(T1/T2/T3) and `Iterator[str]` at the agent layer (T5, where `yield from` consumes the provider generator and
captures `StreamResult`). `_instrument_stream(pieces: Iterator[str]) -> Iterator[str]` (T4). `run_stream ->
Iterator[str]` (T7). `_stream -> Iterator[str]` (T5). `supports_streaming` identity compare against
`BaseAgent._stream` (T5). `_UsageProvider.usage`/`cost` constants (11, 7, 0.25) consistent across T5 and T7.
Consistent throughout.
