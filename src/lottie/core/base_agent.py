"""BaseAgent — LLM-backed, role-driven unit that reasons and decides.

Agents call skills as tools and reason via an injected `LLMProvider`. Every
`run` is auto-instrumented; token and cost usage is captured transparently as
long as LLM calls go through `self.complete`.
"""

from __future__ import annotations

import contextvars
import warnings
from abc import abstractmethod
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel

from lottie.core.metrics import Kind
from lottie.core.runnable import InstrumentedRunnable
from lottie.governance.audit import AuditLogger, build_audit_logger, hash_model
from lottie.governance.cost import BudgetExceeded, CostGate, NullCostGate
from lottie.governance.policy import NullPolicyGate, PolicyEscalation, PolicyGate, PolicyViolation
from lottie.governance.schema import AuditRecord
from lottie.llm import LLMProvider, LLMResponse, Message
from lottie.llm.base import StreamResult
from lottie.memory.base import MemoryClient, NullMemoryClient


class NotStreamable(RuntimeError):
    """Raised if `_stream` is called on an agent that did not opt in."""


# Run depth → the `root` flag (depth 1 = top-level). A ContextVar (not threading.local)
# so the depth propagates into LangGraph parallel worker threads (langgraph copies the
# context when it forks branches), keeping nested workers root=False on any thread.
_audit_depth: contextvars.ContextVar[int] = contextvars.ContextVar("lottie_audit_depth", default=0)


def _depth() -> int:
    return _audit_depth.get()


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
        self._policy: PolicyGate = NullPolicyGate()
        self._cost: CostGate = NullCostGate()

    def set_policy(self, gate: PolicyGate) -> None:
        """Attach a policy gate (called by instantiate_agent for CLI/serve runs)."""
        self._policy = gate

    def set_cost_gate(self, gate: CostGate) -> None:
        """Attach a cost-budget gate (called by instantiate_agent for CLI/serve runs)."""
        self._cost = gate

    @property
    def provider(self) -> str | None:
        return self.llm.model

    def _pre_run_gates(self, data: InputT) -> None:
        """Policy then budget pre-checks; audit a block and re-raise if either trips.

        Shared by run and run_stream.
        """
        try:
            self._policy.check()   # capability policy — checked FIRST (no I/O)
            self._cost.check()     # cumulative budget — checked SECOND (reads the ledger)
        except PolicyViolation as exc:
            self._write_block(
                data, exc, "escalated" if isinstance(exc, PolicyEscalation) else "denied"
            )
            raise
        except BudgetExceeded as exc:
            self._write_block(data, exc, "budget_exceeded")
            raise

    def run(self, data: InputT) -> OutputT:
        """Policy + budget pre-checks, then instrumented run + audit (best-effort)."""
        self._pre_run_gates(data)
        token = _audit_depth.set(_depth() + 1)
        is_root = _depth() == 1
        output: OutputT | None = None
        try:
            output = super().run(data)
            return output
        finally:
            try:
                self._write_audit(data, output, is_root)
            finally:
                _audit_depth.reset(token)

    def run_stream(self, data: InputT) -> Iterator[str]:
        """Streaming analog of run(): same policy/cost pre-gates, instrumented stream, audit post.

        A generator — the pre-gates run on the first `next()`, before any delta, so a
        deny/over-budget raises before the first piece. The output security gate is NOT
        here; it wraps the deltas at the serve boundary (slice 3b), exactly like the
        non-streaming output gate.
        """
        self._pre_run_gates(data)
        token = _audit_depth.set(_depth() + 1)
        is_root = _depth() == 1
        try:
            yield from self._instrument_stream(self._stream(data))
        finally:
            try:
                # output=None: a stream has no single typed Output
                self._write_audit(data, None, is_root)
            finally:
                _audit_depth.reset(token)

    def _write_block(
        self,
        data: InputT,
        exc: Exception,
        status: Literal["denied", "escalated", "budget_exceeded"],
    ) -> None:
        try:
            self._audit.log(
                AuditRecord(
                    ts=datetime.now(UTC).isoformat(),
                    agent=self.name,
                    provider=self.provider,
                    status=status,
                    # pre-check runs before the depth increment, so depth 0 == top-level here
                    root=_depth() == 0,
                    input_sha256=hash_model(data),
                    output_sha256=None,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    latency_ms=0.0,
                    error=str(exc),
                )
            )
        except Exception as e:  # never let auditing convert/suppress the block
            warnings.warn(f"block audit failed: {e}", stacklevel=2)

    def _write_audit(self, data: InputT, output: OutputT | None, is_root: bool) -> None:
        m = self.last_metrics
        if m is None:  # super().run always sets it, but stay defensive
            return
        try:
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
        except Exception as exc:  # never let auditing break a run
            warnings.warn(f"audit record failed: {exc}", stacklevel=2)

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

    def _stream(self, data: InputT) -> Iterator[str]:
        """Opt-in streaming producer. Default raises; override to enable real token streaming."""
        raise NotStreamable(f"{self.name} does not implement _stream")

    @classmethod
    def supports_streaming(cls) -> bool:
        """True if this agent overrides `_stream` (real-stream vs format-fallback in transport)."""
        return cls._stream is not BaseAgent._stream

    def stream_complete(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> Iterator[str]:
        """Stream deltas, accumulating usage into the active run at stream end."""
        result: StreamResult = yield from self.llm.stream_complete(messages, model_params)
        if self._active_ctx is not None:
            self._active_ctx.add_usage(result.usage, result.cost_usd)

    @abstractmethod
    def _execute(self, data: InputT) -> OutputT: ...
