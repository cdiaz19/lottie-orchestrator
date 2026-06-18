from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry")  # skip-guard: needs the [otel] extra

from lottie.governance.otel import run_span, span_set_error, span_set_metrics  # noqa: E402


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


def test_run_span_emits_with_metrics(otel_exporter: object) -> None:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    assert isinstance(otel_exporter, InMemorySpanExporter)
    with run_span("digest", "agent") as span:
        assert span is not None
        span_set_metrics(span, _Metrics())
    spans = otel_exporter.get_finished_spans()
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "digest"
    attrs = dict(s.attributes or {})
    assert attrs["lottie.kind"] == "agent"
    assert attrs["lottie.cost_usd"] == 0.01
    assert attrs["lottie.status"] == "ok"


def test_run_span_nests(otel_exporter: object) -> None:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    assert isinstance(otel_exporter, InMemorySpanExporter)
    with run_span("outer", "agent"):  # noqa: SIM117
        with run_span("inner", "skill") as inner:
            assert inner is not None
    spans = {s.name: s for s in otel_exporter.get_finished_spans()}
    assert spans["inner"].parent is not None
    assert spans["inner"].parent.span_id == spans["outer"].context.span_id


def test_disabled_env_is_noop(otel_exporter: object, monkeypatch: pytest.MonkeyPatch) -> None:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    assert isinstance(otel_exporter, InMemorySpanExporter)
    monkeypatch.setenv("LOTTIE_DISABLE_OTEL", "1")
    with run_span("x", "agent") as span:
        assert span is None
    assert otel_exporter.get_finished_spans() == ()


def test_no_endpoint_is_noop(otel_exporter: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    with run_span("x", "agent") as span:
        assert span is None


def test_error_marks_span_error(otel_exporter: object) -> None:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    assert isinstance(otel_exporter, InMemorySpanExporter)
    with run_span("boom", "agent") as span:
        span_set_error(span, RuntimeError("kaboom"))
    s = otel_exporter.get_finished_spans()[0]
    attrs = dict(s.attributes or {})
    error_val = attrs["lottie.error"]
    assert isinstance(error_val, str) and error_val.startswith("RuntimeError")
    assert s.status.status_code.name == "ERROR"


def test_run_span_fail_open_on_setup_error(
    otel_exporter: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Any tracer-setup failure must degrade to no-span, never raise (fail-open boundary).
    import lottie.governance.otel as otel

    def _boom() -> None:
        raise RuntimeError("tracer dead")

    monkeypatch.setattr(otel, "_ensure_provider", _boom)
    with run_span("x", "agent") as span:  # must NOT raise
        assert span is None
