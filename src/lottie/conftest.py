"""Shared pytest fixtures for lottie.* test modules."""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture
def otel_exporter(monkeypatch: pytest.MonkeyPatch) -> Iterator[object]:
    """A fresh in-memory OTel exporter wired to a per-test TracerProvider.

    Installs the provider by monkeypatching opentelemetry's private `_TRACER_PROVIDER`
    (so monkeypatch auto-restores it on teardown and we sidestep OTel's once-per-process
    `set_tracer_provider` guard — function scope keeps every test isolated). Also turns
    tracing on for the test. Skips when the [otel] extra isn't installed.
    """
    pytest.importorskip("opentelemetry")
    from opentelemetry import trace as _trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    assert hasattr(_trace, "_TRACER_PROVIDER"), (
        "opentelemetry.trace._TRACER_PROVIDER moved — update this fixture"
    )
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(_trace, "_TRACER_PROVIDER", provider)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.delenv("LOTTIE_DISABLE_OTEL", raising=False)
    yield exporter
