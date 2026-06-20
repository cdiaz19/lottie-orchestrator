from __future__ import annotations

from collections.abc import Generator, Iterator

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
    gen: Generator[str, None, None] = stub._instrument_stream(_producer(stub))
    assert next(gen) == "a"
    gen.close()
    m = stub.last_metrics
    assert m is not None and m.success is False
    assert m.error == "stream closed before completion"
    assert m.input_tokens == 3  # usage accumulated before the close is retained
