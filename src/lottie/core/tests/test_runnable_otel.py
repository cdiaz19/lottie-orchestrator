from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry")

from pydantic import BaseModel  # noqa: E402

from lottie.core.base_agent import BaseAgent  # noqa: E402
from lottie.llm import MockLLMProvider  # noqa: E402


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
def _disable_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOTTIE_DISABLE_AUDIT", "1")  # keep audit quiet


def _llm() -> MockLLMProvider:
    return MockLLMProvider(["x"])


def test_run_emits_span_with_metrics(otel_exporter: object) -> None:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    assert isinstance(otel_exporter, InMemorySpanExporter)
    _Inner(_llm(), name="inner").run(_In(q="hi"))
    spans = otel_exporter.get_finished_spans()
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "inner"
    assert s.attributes is not None
    assert s.attributes["lottie.agent"] == "inner"
    assert s.attributes["lottie.status"] == "ok"
    assert "lottie.latency_ms" in s.attributes


def test_nested_run_spans_parent_correctly(otel_exporter: object) -> None:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    assert isinstance(otel_exporter, InMemorySpanExporter)
    _Outer(_llm(), _Inner(_llm(), name="inner")).run(_In(q="hi"))
    spans = {s.name: s for s in otel_exporter.get_finished_spans()}
    assert spans["inner"].parent is not None
    assert spans["inner"].parent.span_id == spans["_Outer"].context.span_id


def test_failed_run_marks_span_error_and_reraises(otel_exporter: object) -> None:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    assert isinstance(otel_exporter, InMemorySpanExporter)
    with pytest.raises(RuntimeError):
        _Boom(_llm(), name="boom").run(_In(q="hi"))
    s = otel_exporter.get_finished_spans()[0]
    assert s.attributes is not None
    assert s.attributes["lottie.status"] == "error"
    assert s.status.status_code.name == "ERROR"
