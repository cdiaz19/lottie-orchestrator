"""BaseAgent — LLM-backed, role-driven unit that reasons and decides.

Agents call skills as tools and reason via an injected `LLMProvider`. Every
`run` is auto-instrumented; token and cost usage is captured transparently as
long as LLM calls go through `self.complete`.
"""

from __future__ import annotations

import threading
import warnings
from abc import abstractmethod
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from lottie.core.metrics import Kind
from lottie.core.runnable import InstrumentedRunnable
from lottie.governance.audit import AuditLogger, build_audit_logger, hash_model
from lottie.governance.policy import NullPolicyGate, PolicyEscalation, PolicyGate, PolicyViolation
from lottie.governance.schema import AuditRecord
from lottie.llm import LLMProvider, LLMResponse, Message
from lottie.memory.base import MemoryClient, NullMemoryClient

# Per-thread run depth → the `root` flag (depth 1 = top-level). NOTE: this assumes a
# nested run shares the orchestrator's thread (LocalEngine / sequential mesh). A
# LangGraph parallel worker runs on its own thread and will be flagged root=True.
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
        self._policy: PolicyGate = NullPolicyGate()

    def set_policy(self, gate: PolicyGate) -> None:
        """Attach a policy gate (called by instantiate_agent for CLI/serve runs)."""
        self._policy = gate

    @property
    def provider(self) -> str | None:
        return self.llm.model

    def run(self, data: InputT) -> OutputT:
        """Policy pre-check, then instrumented run + audit (best-effort)."""
        try:
            self._policy.check()
        except PolicyViolation as exc:
            self._write_policy_block(data, exc)
            raise
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

    def _write_policy_block(self, data: InputT, exc: PolicyViolation) -> None:
        status = "escalated" if isinstance(exc, PolicyEscalation) else "denied"
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
        except Exception as e:  # never let auditing convert/suppress the policy block
            warnings.warn(f"policy-block audit failed: {e}", stacklevel=2)

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

    @abstractmethod
    def _execute(self, data: InputT) -> OutputT: ...
