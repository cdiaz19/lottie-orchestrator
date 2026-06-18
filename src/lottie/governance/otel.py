"""OpenTelemetry tracing for runs. Opt-in [otel] extra; no-op when absent/unconfigured;
fail-open (no tracer or collector failure ever breaks or blocks a run). Imports only
stdlib + guarded opentelemetry — no core/project deps (acyclic).
"""

from __future__ import annotations

import contextlib
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
        with contextlib.suppress(Exception):
            span.set_attribute("lottie.kind", kind)
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
