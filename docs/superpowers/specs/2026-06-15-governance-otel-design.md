# Governance — OpenTelemetry Tracing (slice 4) — Design

> Emit one OpenTelemetry span per run — for **every** agent and skill — wrapping the existing
> instrumented `run`. Opt-in `[otel]` extra, no-op when absent or unconfigured, **fail-open**
> (a dead collector never breaks a run), privacy-safe (attributes only, no raw payloads).

- **Date:** 2026-06-15
- **Phase:** Governance, slice 4 (after audit #11, policy #12, cost #13 — all on `main`).
- **Branch:** `feat/governance-otel` (off the freshly-merged `main`).

---

## 1. Goal & scope

Per-run latency/tokens/cost are already captured (`RunMetrics`) and the audit trail records them, but
there's no distributed-trace view: no spans, no parent/child run tree, no OTLP export to a collector
(Jaeger/Tempo/etc.). This slice adds OpenTelemetry tracing: a span per run, nested into the natural
agent→skill and mesh→worker tree, exported via OTLP when configured.

**Locked decisions (do not relitigate):**
- **Span seam:** `InstrumentedRunnable.run` (`core/runnable.py`) — **every** run, agents AND skills
  (not `BaseAgent.run`). Sits next to the existing `_record`; reads the already-captured
  `RunContext`/`last_metrics`, emits — does not duplicate.
- **Nesting:** via OTel's own context propagation (contextvars-backed — the same mechanism class as the
  audit root-flag fix `e99d42e`). Same-thread nesting (agent→skill, sequential mesh) is guaranteed and
  tested. **Cross-thread parallel-mesh nesting is *expected but unverified*** (relies on langgraph
  copying the OTel context into its worker threads, as it does for the audit ContextVar) — validated in
  lab Round 9; if it doesn't hold it moves to §7 deferred.
- **Opt-in `[otel]` extra** (like `[mesh]`/`[serve]`): base install OTel-free, all OTel tests
  skip-guarded; OTLP exporter, env-driven endpoint, **no-op when the extra is absent or no endpoint is
  set**.
- **Fail-open / best-effort:** observability, not enforcement — a dead/slow collector or any tracer
  error must NEVER break or slow a run beyond the async exporter.
- **Attributes:** `name`, `kind`, `status`, `latency_ms`, `input_tokens`, `output_tokens`, `cost_usd`,
  `provider`, `error`. **Privacy: NO raw payloads** (parity with the audit trail's sha256-only posture).
- **Grain differs from audit on purpose:** audit is agent-level (`BaseAgent.run`); tracing is every run
  (agents + skills, `InstrumentedRunnable.run`). This asymmetry is intentional — tracing's value is the
  full tree; noted here so a future reader doesn't "align" them.

## 2. The `[otel]` extra & no-op model

`pyproject.toml` gains an optional-dependency group `otel = ["opentelemetry-sdk", "opentelemetry-exporter-otlp"]`
(mirroring `[mesh]`). Base install pulls nothing. At runtime the tracer is resolved once:

- **opentelemetry not importable** (extra absent) → a **no-op** span context manager (zero overhead).
- **importable but no exporter endpoint configured** (the standard `OTEL_EXPORTER_OTLP_ENDPOINT`
  unset) → we treat "no endpoint" as "tracing off" and return the no-op CM, avoiding any provider/
  exporter setup cost. Setting `OTEL_EXPORTER_OTLP_ENDPOINT` (the canonical OTLP env) turns it on.

A `LOTTIE_DISABLE_OTEL` env (truthy) forces no-op even when configured — symmetry with
`LOTTIE_DISABLE_AUDIT`, and the lever the test suite uses to stay quiet.

## 3. `governance/otel.py` — tracer + span helper

```python
@contextmanager
def run_span(name: str, kind: str) -> Iterator[_Span | None]:
    """Yield an active span for a run, or None when tracing is off. Never raises."""
```

- `run_span` is the ONLY thing `core/runnable.py` imports. It resolves the tracer lazily + memoized
  (`_tracer()` builds a `TracerProvider` + `BatchSpanProcessor(OTLPSpanExporter())` on first use when
  enabled; returns `None` when disabled/absent), then `tracer.start_as_current_span(name)`.
- **Fail-open is total:** the whole body is guarded so that import errors, provider/exporter
  construction failures, or `start_as_current_span` raising all degrade to yielding `None` (no span),
  never propagating. `BatchSpanProcessor` exports off-thread, so a dead collector can't block the run.
- Helpers `span_set_metrics(span, metrics)` and `span_set_error(span, exc)` set attributes / error
  status, each a no-op when `span is None` and each individually guarded (best-effort).
- `governance/otel.py` imports only stdlib + (guarded) `opentelemetry` — **no `core`/`project`**, so
  `core.runnable → governance.otel` stays acyclic (same direction as `core.base_agent → governance.audit`).

> Module placement: in `governance/` to group the four governance slices (audit/policy/cost/otel),
> hooked from core — exactly how audit is hooked from `base_agent`. The `core → governance` edge already
> exists and is acyclic.

## 4. The hook — `InstrumentedRunnable.run` (`src/lottie/core/runnable.py`)

Wrap `_execute` in `run_span`; set attributes from `self.last_metrics` after `_record` (so
tokens/cost/latency are populated); set error status on exception. The existing
`RunContext`/`_record`/`append_metrics` flow is unchanged.

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

- The span wraps `_execute`, so a nested run (a skill called inside an agent's `_execute`, or a mesh
  worker inside the engine) opens its span while the parent span is the current OTel context → it nests
  automatically. Mesh **parallel** workers are *expected* to nest via the same context copy langgraph
  performs into its worker threads (the property the audit root-flag fix `e99d42e` relied on) — OTel's
  context is contextvars-backed, like the audit depth. **Unverified in this slice** (only same-thread
  agent→skill nesting is tested); OTel context does not auto-propagate into arbitrary threads without
  `copy_context`, so whether langgraph's copy carries the active span is an open question — **validated
  in lab Round 9**, with cross-thread parent-link propagation listed under §7 deferred if it doesn't hold.
- `span_set_metrics` reads `last_metrics` (set by `_record` in the same `finally`): sets
  `lottie.agent`/`lottie.kind`/`lottie.status`/`lottie.latency_ms`/`lottie.input_tokens`/
  `lottie.output_tokens`/`lottie.cost_usd`/`lottie.provider`. `span_set_error` sets the OTel error
  status + an `error` attribute = `repr(exc)` (an exception repr, never the input/output payload).

## 5. Attributes & privacy

Span name = the runnable `name`. Attributes are the same scalar fields the audit trail already deems
safe (status/tokens/cost/provider/latency) — **never** `data` or the output. No `model_dump`, no hashes
even; this is a metrics-shaped trace, not a content store. Error attribute is the exception `repr`
(same as `RunMetrics.error`); an exception message *could* echo input, accepted exactly as in the audit
slice.

## 6. Testing

All OTel tests skip-guarded (`pytest.importorskip("opentelemetry")` / the `[mesh]`-style
`_HAS_OTEL` flag).

- **`run_span` / helpers** (unit, with an in-memory exporter): configuring an
  `InMemorySpanExporter` + `SimpleSpanProcessor` (test-local `TracerProvider`), a run produces one span
  named after the runnable with the expected attributes; `LOTTIE_DISABLE_OTEL` → no span; extra absent
  → `run_span` yields `None` and the run still returns.
- **Nesting** (integration): an agent whose `_execute` runs a second agent (same thread) → the inner
  span's parent is the outer span (assert `parent.span_id`). Parallel LangGraph mesh nesting is NOT
  unit-tested here (its own thread + the `[mesh]` extra) — checked in lab Round 9.
- **Fail-open** (unit): a tracer/exporter that raises on span start or on attribute-set must NOT break
  the run — monkeypatch the tracer to raise, assert the run still returns its output and no exception
  leaks; a run that itself raises still propagates its OWN exception (not a tracing error), with the
  span marked error.
- **Base-install safety:** `import lottie.core.runnable` and a normal agent run work with opentelemetry
  NOT installed (the no-op path) — guarded so CI's base (no `[otel]`) job stays green.
- **Full gate at closeout:** `uv run pytest -q`, `uv run mypy --strict src`, `uv run ruff check` — all
  green; the existing ~756 tests unaffected (no endpoint set ⇒ no-op everywhere). Every test helper
  gets a return-type annotation (mypy --strict).

## 7. Out of scope / deferred (YAGNI)

- **Per-`complete()` LLM child spans** — a span around each `self.complete` call. Valuable, but touches
  the `complete()`/provider path and is tangled with FU-2 (skill-internal token accounting); its own
  slice.
- **Denied/blocked-run spans** — policy/cost/budget denials raise in `BaseAgent.run` **before**
  `super().run()` (= `InstrumentedRunnable.run`), so v1 traces **executed** runs only. Denials are still
  recorded in the audit ledger (`status="denied"/"escalated"/"budget_exceeded"`); a future slice can add
  a denial span at the `BaseAgent.run` pre-check if desired.
- **Metrics/logs signals, baggage, custom samplers, span links, resource attributes beyond service
  name** — OTLP trace export with default sampling only.

## 8. Definition of done

Every run (agent + skill) emits one OTel span wrapping `_execute`, with the scalar attributes (no raw
payloads), nested into the agent→skill / sequential mesh tree (parallel-worker nesting expected but
validated in Round 9); opt-in `[otel]`
extra with a no-op default (absent / no endpoint / `LOTTIE_DISABLE_OTEL`); **fail-open** — no tracer or
collector failure can break or block a run; OTLP export when `OTEL_EXPORTER_OTLP_ENDPOINT` is set;
`governance.otel` acyclic; base install OTel-free with all OTel tests skip-guarded. `uv run pytest -q` /
`mypy --strict src` / `ruff check` green. Validate downstream in lottie-lab Round 9 before merging.
Commit on the feature branch; do not push until approved.
