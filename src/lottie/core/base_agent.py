"""BaseAgent — LLM-backed, role-driven unit that reasons and decides.

Agents call skills as tools and reason via an injected `LLMProvider`. Every
`run` is auto-instrumented; token and cost usage is captured transparently as
long as LLM calls go through `self.complete`.
"""

from __future__ import annotations

import threading
from abc import abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from lottie.core.metrics import Kind
from lottie.core.runnable import InstrumentedRunnable
from lottie.governance.audit import AuditLogger, build_audit_logger, hash_model
from lottie.governance.schema import AuditRecord
from lottie.llm import LLMProvider, LLMResponse, Message
from lottie.memory.base import MemoryClient, NullMemoryClient

_audit_depth = threading.local()


def _depth() -> int:
    return getattr(_audit_depth, "value", 0)


class BaseAgent[InputT: BaseModel, OutputT: BaseModel](InstrumentedRunnable[InputT, OutputT]):
    """Extend this for every agent. Implement only `_execute`."""

    kind: ClassVar[Kind] = "agent"

    def __init__(
        self,
        llm: LLMProvider,
        *,
        name: str | None = None,
        memory: MemoryClient | None = None,
        enable_benchmarks: bool | None = None,
        benchmarks_root: Path | None = None,
        audit: AuditLogger | None = None,
    ) -> None:
        super().__init__(
            name=name,
            enable_benchmarks=enable_benchmarks,
            benchmarks_root=benchmarks_root,
        )
        self.llm = llm
        self.memory: MemoryClient = memory or NullMemoryClient()
        self._audit = audit if audit is not None else build_audit_logger(self._benchmarks_root)

    @property
    def provider(self) -> str | None:
        return self.llm.model

    def run(self, data: InputT) -> OutputT:
        """Instrumented run (super) plus one immutable audit record (best-effort)."""
        _audit_depth.value = _depth() + 1
        is_root = _depth() == 1
        output: OutputT | None = None
        try:
            output = super().run(data)
            return output
        finally:
            try:
                self._write_audit(data, output, is_root)
            finally:
                _audit_depth.value = _depth() - 1

    def _write_audit(self, data: InputT, output: OutputT | None, is_root: bool) -> None:
        m = self.last_metrics
        if m is None:  # super().run always sets it, but stay defensive
            return
        record = AuditRecord(
            ts=m.timestamp.isoformat(),
            agent=m.name,
            provider=m.provider,
            status="ok" if m.success else "error",
            root=is_root,
            input_sha256=hash_model(data),
            output_sha256=hash_model(output) if output is not None else None,
            input_tokens=m.input_tokens,
            output_tokens=m.output_tokens,
            cost_usd=m.cost_usd,
            latency_ms=m.latency_ms,
            error=m.error,
        )
        self._audit.log(record)

    def complete(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> LLMResponse:
        """Run an LLM completion, accumulating tokens/cost into the active run."""
        response = self.llm.complete(messages, model_params)
        if self._active_ctx is not None:
            self._active_ctx.add_usage(response.usage, response.cost_usd)
        return response

    @abstractmethod
    def _execute(self, data: InputT) -> OutputT: ...
