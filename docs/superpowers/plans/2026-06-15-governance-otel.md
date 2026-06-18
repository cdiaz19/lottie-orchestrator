# Governance OpenTelemetry Tracing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit one OpenTelemetry span per run (every agent AND skill) wrapping the existing instrumented `run`, opt-in via an `[otel]` extra, no-op when absent/unconfigured, fail-open (no tracer/collector failure breaks a run), privacy-safe (scalar attributes only).

**Architecture:** `governance/otel.py` exposes `run_span` (a context manager yielding a span or `None`) + `span_set_metrics`/`span_set_error`, all fail-open. `InstrumentedRunnable.run` (the shared agent+skill seam) wraps `_execute` in `run_span` and sets attributes from the already-recorded `last_metrics`. Nesting is automatic via OTel's contextvars-backed context (propagates into LangGraph parallel workers, like the audit root-flag fix).

**Tech Stack:** Python 3.12, OpenTelemetry SDK **1.42.x** (pinned below), pytest, mypy --strict, ruff. Branch `feat/governance-otel` off `main` (already checked out). Tools via `uv run`.

**Pinned OTel API (verified against the installed 1.42.x wheel — use these exact imports):**
- `from opentelemetry import trace` → `trace.get_tracer("lottie")`, `trace.get_tracer_provider()`, `trace.set_tracer_provider(provider)`.
- `from opentelemetry.trace import Status, StatusCode` → `StatusCode.OK` / `StatusCode.ERROR`; `span.set_status(Status(StatusCode.ERROR))`.
- `tracer.start_as_current_span(name)` is a context manager → `with ... as span:`; `span.set_attribute(key, value)`.
- `from opentelemetry.sdk.trace import TracerProvider`; `from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor`; `from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter` (`.get_finished_spans()` → spans with `.name`, `.attributes` (mapping), `.parent` (a `SpanContext` or `None`)); `from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter`; `from opentelemetry.sdk.resources import Resource`.
- **OTel constraint:** `trace.set_tracer_provider` only takes effect ONCE per process. Tests therefore install a module-level `TracerProvider` + `InMemorySpanExporter` once and call `exporter.clear()` per test.

**Key facts (verified):**
- `InstrumentedRunnable.run` (`core/runnable.py`): builds `RunContext`, times, `try: return self._execute(data) except: success=False; error=repr(exc); raise finally: self._record(ctx, start, success, error); self._active_ctx=None`. `_record` sets `self.last_metrics` (a `RunMetrics`: `name, kind, provider, latency_ms, input_tokens, output_tokens, cost_usd, success, error`).
- `pyproject.toml` has `[project.optional-dependencies]` with a `mesh = [...]` group to mirror.
- `governance/otel.py` must import only stdlib + guarded `opentelemetry` (no `core`/`project`) — acyclic, like `governance.audit`.

---

### Task 1: `[otel]` extra + `governance/otel.py`

**Files:**
- Modify: `pyproject.toml` (add the `otel` optional-dependency group)
- Create: `src/lottie/governance/otel.py`
- Test: `src/lottie/governance/tests/test_otel.py`

- [ ] **Step 1: Add the extra** to `pyproject.toml` under `[project.optional-dependencies]`, next to `mesh`:
```toml
otel = [
    "opentelemetry-sdk>=1.42",
    "opentelemetry-exporter-otlp-proto-http>=1.42",
]
```
Then `uv pip install -e '.[otel]'` so the dev env has it (it is already installed from the spec's pinning step, but make the extra authoritative).

- [ ] **Step 2: Write the failing test** — `src/lottie/governance/tests/test_otel.py`:
```python
from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry")  # skip-guard: needs the [otel] extra

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from lottie.governance.otel import run_span, span_set_error, span_set_metrics  # noqa: E402

# Install one global provider + in-memory exporter for the whole module (set_tracer_provider
# only takes effect once per process); clear the exporter between tests.
_EXPORTER = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))
trace.set_tracer_provider(_provider)


class _Metrics:
    name = "digest"
    kind = "agent"
    provider = "mock/x"
    latency_ms = 12.5
    input_tokens = 10
    output_tokens = 20
    cost_usd = 0.01
    success = True
    error = None


@pytest.fixture(autouse=True)
def _trace_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.delenv("LOTTIE_DISABLE_OTEL", raising=False)
    _EXPORTER.clear()


def test_run_span_emits_with_metrics() -> None:
    with run_span("digest", "agent") as span:
        assert span is not None
        span_set_metrics(span, _Metrics())
    spans = _EXPORTER.get_finished_spans()
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "digest"
    assert s.attributes["lottie.kind"] == "agent"
    assert s.attributes["lottie.cost_usd"] == 0.01
    assert s.attributes["lottie.status"] == "ok"


def test_run_span_nests() -> None:
    with run_span("outer", "agent"):
        with run_span("inner", "skill") as inner:
            assert inner is not None
    spans = {s.name: s for s in _EXPORTER.get_finished_spans()}
    assert spans["inner"].parent is not None
    assert spans["inner"].parent.span_id == spans["outer"].context.span_id


def test_disabled_env_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOTTIE_DISABLE_OTEL", "1")
    with run_span("x", "agent") as span:
        assert span is None
    assert _EXPORTER.get_finished_spans() == ()


def test_no_endpoint_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    with run_span("x", "agent") as span:
        assert span is None


def test_error_marks_span_error() -> None:
    with run_span("boom", "agent") as span:
        span_set_error(span, RuntimeError("kaboom"))
    s = _EXPORTER.get_finished_spans()[0]
    assert s.attributes["lottie.error"].startswith("RuntimeError")
    assert s.status.status_code.name == "ERROR"


def test_run_span_fail_open_on_setup_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any tracer-setup failure must degrade to no-span, never raise (fail-open boundary).
    import lottie.governance.otel as otel

    def _boom() -> None:
        raise RuntimeError("tracer dead")

    monkeypatch.setattr(otel, "_ensure_provider", _boom)
    with run_span("x", "agent") as span:  # must NOT raise
        assert span is None
```
This is THE fail-open boundary test: `run_span` is guaranteed non-raising by construction (the
`try/except → yield None` around `_ensure_provider()` + `start_as_current_span`). Because `run_span`
can't raise, the run path that uses it (Task 2) is fail-open for free — no separate run-level guard or
dead-collector test is needed (and a real dead collector is async via `BatchSpanProcessor`, so it can't
block a run either).

- [ ] **Step 3: Run, verify FAIL** — `uv run pytest src/lottie/governance/tests/test_otel.py -v` → `ModuleNotFoundError: lottie.governance.otel`.

- [ ] **Step 4: Implement `src/lottie/governance/otel.py`:**
```python
"""OpenTelemetry tracing for runs. Opt-in [otel] extra; no-op when absent/unconfigured;
fail-open (no tracer or collector failure ever breaks or blocks a run). Imports only
stdlib + guarded opentelemetry — no core/project deps (acyclic).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_DISABLE = {"1", "true", "yes", "on"}

try:
    from opentelemetry import trace as _trace
    from opentelemetry.trace import Status as _Status
    from opentelemetry.trace import StatusCode as _StatusCode

    _HAS_OTEL = True
except ImportError:  # base install without the [otel] extra
    _HAS_OTEL = False


def _enabled() -> bool:
    if not _HAS_OTEL:
        return False
    if os.getenv("LOTTIE_DISABLE_OTEL", "").lower() in _DISABLE:
        return False
    return bool(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))


def _ensure_provider() -> None:
    """Install the OTLP TracerProvider once. Respect a provider already set (e.g. by tests)."""
    from opentelemetry.sdk.trace import TracerProvider

    if isinstance(_trace.get_tracer_provider(), TracerProvider):
        return  # a real provider is already installed — don't override it
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "lottie"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    _trace.set_tracer_provider(provider)


@contextmanager
def run_span(name: str, kind: str) -> Iterator[Any]:
    """Yield an active span for a run, or None when tracing is off. Never raises."""
    if not _enabled():
        yield None
        return
    try:
        _ensure_provider()
        cm = _trace.get_tracer("lottie").start_as_current_span(name)
    except Exception:  # noqa: BLE001 — fail-open: any tracer setup failure => no span
        yield None
        return
    with cm as span:
        try:
            span.set_attribute("lottie.kind", kind)
        except Exception:  # noqa: BLE001
            pass
        yield span


def span_set_metrics(span: Any, metrics: Any) -> None:
    """Set scalar run attributes + status from RunMetrics. No raw payloads. Best-effort."""
    if span is None or metrics is None:
        return
    try:
        span.set_attribute("lottie.agent", metrics.name)
        span.set_attribute("lottie.kind", metrics.kind)
        span.set_attribute("lottie.status", "ok" if metrics.success else "error")
        span.set_attribute("lottie.latency_ms", metrics.latency_ms)
        span.set_attribute("lottie.input_tokens", metrics.input_tokens)
        span.set_attribute("lottie.output_tokens", metrics.output_tokens)
        span.set_attribute("lottie.cost_usd", metrics.cost_usd)
        if metrics.provider is not None:
            span.set_attribute("lottie.provider", metrics.provider)
        if not metrics.success:
            span.set_status(_Status(_StatusCode.ERROR))
    except Exception:  # noqa: BLE001
        pass


def span_set_error(span: Any, exc: BaseException) -> None:
    """Mark the span errored (exception repr only — never the payload). Best-effort."""
    if span is None:
        return
    try:
        span.set_attribute("lottie.error", repr(exc))
        span.set_status(_Status(_StatusCode.ERROR))
    except Exception:  # noqa: BLE001
        pass
```

- [ ] **Step 5: Run, verify PASS** — `uv run pytest src/lottie/governance/tests/test_otel.py -v` → all pass. (If ruff flags `# noqa: BLE001` as unused, remove the comments but KEEP the broad `except`.)

- [ ] **Step 6: Gates** — `uv run mypy --strict src/lottie/governance && uv run ruff check src/lottie/governance` → clean. (`Any` is used deliberately for the OTel span/tracer — they're typed `Any` because the SDK types are only importable under the extra; the broad-catch fail-open also justifies it.)

- [ ] **Step 7: Commit**
```bash
git add pyproject.toml src/lottie/governance/otel.py src/lottie/governance/tests/test_otel.py
git commit -m "feat(governance): OpenTelemetry tracer + run_span (opt-in [otel], fail-open, no-op default)"
```

---

### Task 2: Hook the span into `InstrumentedRunnable.run`

**Files:**
- Modify: `src/lottie/core/runnable.py`
- Test: `src/lottie/core/tests/test_runnable_otel.py`

- [ ] **Step 1: Write the failing test** — `src/lottie/core/tests/test_runnable_otel.py`:
```python
from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry")

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from pydantic import BaseModel  # noqa: E402

from lottie.core.base_agent import BaseAgent  # noqa: E402
from lottie.llm import MockLLMProvider  # noqa: E402

_EXPORTER = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))
trace.set_tracer_provider(_provider)


class _In(BaseModel):
    q: str


class _Out(BaseModel):
    a: str


class _Inner(BaseAgent[_In, _Out]):
    def _execute(self, data: _In) -> _Out:
        return _Out(a=f"inner:{data.q}")


class _Outer(BaseAgent[_In, _Out]):
    def __init__(self, llm: object, inner: _Inner) -> None:
        super().__init__(llm)  # type: ignore[arg-type]
        self._inner = inner

    def _execute(self, data: _In) -> _Out:
        return self._inner.run(data)


class _Boom(BaseAgent[_In, _Out]):
    def _execute(self, data: _In) -> _Out:
        raise RuntimeError("boom")


@pytest.fixture(autouse=True)
def _trace_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.delenv("LOTTIE_DISABLE_OTEL", raising=False)
    monkeypatch.setenv("LOTTIE_DISABLE_AUDIT", "1")  # keep audit quiet
    _EXPORTER.clear()


def _llm() -> MockLLMProvider:
    return MockLLMProvider(["x"])


def test_run_emits_span_with_metrics() -> None:
    _Inner(_llm(), name="inner").run(_In(q="hi"))
    spans = _EXPORTER.get_finished_spans()
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "inner"
    assert s.attributes["lottie.agent"] == "inner"
    assert s.attributes["lottie.status"] == "ok"
    assert "lottie.latency_ms" in s.attributes


def test_nested_run_spans_parent_correctly() -> None:
    _Outer(_llm(), _Inner(_llm(), name="inner")).run(_In(q="hi"))
    spans = {s.name: s for s in _EXPORTER.get_finished_spans()}
    assert spans["inner"].parent is not None
    assert spans["inner"].parent.span_id == spans["_Outer"].context.span_id


def test_failed_run_marks_span_error_and_reraises() -> None:
    with pytest.raises(RuntimeError):
        _Boom(_llm(), name="boom").run(_In(q="hi"))
    s = _EXPORTER.get_finished_spans()[0]
    assert s.attributes["lottie.status"] == "error"
    assert s.status.status_code.name == "ERROR"
```
(Run-level fail-open needs no test here: `run_span` is non-raising by construction — proven by
`test_run_span_fail_open_on_setup_error` in Task 1 — so `with run_span(...)` in `runnable.run` cannot
break a run. The `_Boom` test confirms the run's OWN exception still propagates with the span marked
error.)

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest src/lottie/core/tests/test_runnable_otel.py -v` → no spans (hook not added).

- [ ] **Step 3: Implement the hook in `src/lottie/core/runnable.py`**

Add the import (with the existing `from lottie.core.metrics import ...`):
```python
from lottie.governance.otel import run_span, span_set_error, span_set_metrics
```
Wrap `_execute` in the span (the `RunContext`/`_record`/`append_metrics` flow is unchanged):
```python
    def run(self, data: InputT) -> OutputT:
        ctx = RunContext()
        self._active_ctx = ctx
        start = perf_counter()
        success = True
        error: str | None = None
        with run_span(self.name, self.kind) as span:
            try:
                return self._execute(data)
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

- [ ] **Step 4: Run the OTel tests + the whole suite** — `uv run pytest src/lottie/core/tests/test_runnable_otel.py -v` (pass), then `uv run pytest -q` (whole suite green — no `OTEL_EXPORTER_OTLP_ENDPOINT` set outside the OTel test modules ⇒ `run_span` is a no-op everywhere ⇒ existing ~756 tests unaffected). If something breaks, STOP and report.

- [ ] **Step 5: Base-install no-op sanity** — confirm the no-op path: `LOTTIE_DISABLE_OTEL=1 uv run pytest src/lottie/core/tests/test_base_agent_audit.py -q` (an existing agent-run test) still passes with tracing forced off. (Simulates the base install without `[otel]`.)

- [ ] **Step 6: Gates** — `uv run mypy --strict src && uv run ruff check` → clean.

- [ ] **Step 7: Commit**
```bash
git add src/lottie/core/runnable.py src/lottie/core/tests/test_runnable_otel.py
git commit -m "feat(core): emit an OTel span per run wrapping _execute (fail-open, no-op default)"
```

---

## Self-review checklist (controller, before finishing)

- [ ] Spec coverage: span per run at `InstrumentedRunnable.run` (agents + skills); attributes name/kind/status/latency/tokens/cost/provider/error, NO raw payloads; nesting via OTel context (agent→skill, parallel mesh); opt-in `[otel]` extra + no-op (absent / no endpoint / `LOTTIE_DISABLE_OTEL`); fail-open; `governance.otel` acyclic; grain-asymmetry intentional.
- [ ] Fail-open is real: `run_span` never raises (guaranteed by construction), and a dead OTLP endpoint doesn't block a run (async `BatchSpanProcessor`). The dead-endpoint test (Task 2 option (a)) proves it.
- [ ] Whole suite green; OTel tests skip-guarded; base path (no endpoint) is a no-op so existing tests are unaffected. Every test helper has a return-type annotation.
- [ ] Type/name consistency: `run_span`, `span_set_metrics`, `span_set_error`, `_enabled`, `_ensure_provider`, `lottie.*` attribute keys.
- [ ] `uv run pytest -q`, `uv run mypy --strict src`, `uv run ruff check` green (FULL gate, not just per-task).
- [ ] Final opus whole-branch review before finishing. Do NOT push — validate in lottie-lab Round 9 first, then PR.
```
