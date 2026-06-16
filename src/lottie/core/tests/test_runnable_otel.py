from __future__ import annotations

from collections.abc import Generator

import pytest

pytest.importorskip("opentelemetry")

import opentelemetry.trace as _trace_mod  # noqa: E402
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
def _trace_on(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.delenv("LOTTIE_DISABLE_OTEL", raising=False)
    monkeypatch.setenv("LOTTIE_DISABLE_AUDIT", "1")  # keep audit quiet
    _EXPORTER.clear()
    # OTel enforces singleton set_tracer_provider; swap via internal API so we own
    # the active provider for our tests and restore the previous one on teardown.
    _prev = _trace_mod._TRACER_PROVIDER
    _trace_mod._TRACER_PROVIDER = _provider
    yield
    _trace_mod._TRACER_PROVIDER = _prev


def _llm() -> MockLLMProvider:
    return MockLLMProvider(["x"])


def test_run_emits_span_with_metrics() -> None:
    _Inner(_llm(), name="inner").run(_In(q="hi"))
    spans = _EXPORTER.get_finished_spans()
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "inner"
    assert s.attributes is not None
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
    assert s.attributes is not None
    assert s.attributes["lottie.status"] == "error"
    assert s.status.status_code.name == "ERROR"
