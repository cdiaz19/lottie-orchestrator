"""Shared template-method base for instrumented agents and skills.

Subclasses implement only `_execute`. The public `run` wraps it with timing,
token/cost accumulation, and benchmark persistence — zero extra code in the
subclass.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterator
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import ClassVar

from pydantic import BaseModel

from lottie.core.metrics import Kind, RunContext, RunMetrics, append_metrics, git_version
from lottie.governance.otel import run_span, span_set_error, span_set_metrics

_DISABLE_VALUES = {"1", "true", "yes", "on"}


def _benchmarks_enabled(explicit: bool | None) -> bool:
    if explicit is not None:
        return explicit
    return os.getenv("LOTTIE_DISABLE_BENCHMARKS", "").lower() not in _DISABLE_VALUES


class InstrumentedRunnable[InputT: BaseModel, OutputT: BaseModel](ABC):
    """Base class providing an instrumented, template-method `run`."""

    kind: ClassVar[Kind]

    def __init__(
        self,
        *,
        name: str | None = None,
        enable_benchmarks: bool | None = None,
        benchmarks_root: Path | None = None,
    ) -> None:
        self.name = name or type(self).__name__
        self._enable_benchmarks = _benchmarks_enabled(enable_benchmarks)
        self._benchmarks_root = benchmarks_root or Path.cwd()
        self.last_metrics: RunMetrics | None = None
        self._active_ctx: RunContext | None = None

    @property
    def provider(self) -> str | None:
        """Provider id for metrics. None for non-LLM runnables."""
        return None

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

    def _record(
        self,
        ctx: RunContext,
        start: float,
        success: bool,
        error: str | None,
    ) -> None:
        record = RunMetrics(
            name=self.name,
            kind=self.kind,
            provider=self.provider,
            timestamp=datetime.now(UTC),
            latency_ms=(perf_counter() - start) * 1000,
            input_tokens=ctx.input_tokens,
            output_tokens=ctx.output_tokens,
            cost_usd=ctx.cost_usd,
            retry_count=ctx.retry_count,
            success=success,
            version=git_version(),
            error=error,
        )
        self.last_metrics = record
        append_metrics(record, self._benchmarks_root, self._enable_benchmarks)

    def _instrument_stream(self, pieces: Iterator[str]) -> Generator[str, None, None]:
        """run()'s instrumentation, streamed: time the run, record metrics post.

        `pieces` is the producer (an agent's `_stream`); it accumulates usage into
        `_active_ctx` as it runs. On early consumer close (`GeneratorExit`) the run is
        recorded PARTIAL, not a clean success — `_record` only writes (never yields),
        so it is safe during close (unlike a flushing generator).
        """
        ctx = RunContext()
        self._active_ctx = ctx
        start = perf_counter()
        success = True
        error: str | None = None
        with run_span(self.name, self.kind) as span:
            try:
                yield from pieces
            except GeneratorExit:  # BaseException, not Exception — handled explicitly, re-raised
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

    @abstractmethod
    def _execute(self, data: InputT) -> OutputT:
        """Subclass logic. Receives typed input, returns typed output."""
