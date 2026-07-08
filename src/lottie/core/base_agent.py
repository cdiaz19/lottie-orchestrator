"""BaseAgent — LLM-backed, role-driven unit that reasons and decides.

Agents call skills as tools and reason via an injected `LLMProvider`. Every
`run` is auto-instrumented; token and cost usage is captured transparently as
long as LLM calls go through `self.complete`.
"""

from __future__ import annotations

import contextvars
import warnings
from abc import abstractmethod
from collections.abc import Generator, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel

from lottie.core.metrics import Kind
from lottie.core.runnable import InstrumentedRunnable
from lottie.core.security_gate import NullSecurityGate, SecurityGateProtocol
from lottie.governance.audit import AuditLogger, build_audit_logger, hash_model
from lottie.governance.capability import (
    CapabilityGate,
    NullCapabilityGate,
    _active_capabilities,
)
from lottie.governance.cost import BudgetExceeded, CostGate, NullCostGate, TokenCapExceeded
from lottie.governance.policy import NullPolicyGate, PolicyEscalation, PolicyGate, PolicyViolation
from lottie.governance.schema import AuditRecord
from lottie.llm import LLMProvider, LLMResponse, Message
from lottie.llm.base import StreamResult
from lottie.memory.base import MemoryClient, NullMemoryClient


class NotStreamable(RuntimeError):
    """Raised if `_stream` is called on an agent that did not opt in."""


class TurnLimitExceeded(RuntimeError):
    """A run made more LLM completions than its `max_turns` cap (runaway-loop guard)."""


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
        self._capabilities: CapabilityGate = NullCapabilityGate()
        self._security: SecurityGateProtocol = NullSecurityGate()
        self._max_run_tokens: int | None = None
        self._max_turns: int | None = None

    def set_policy(self, gate: PolicyGate) -> None:
        """Attach a policy gate (called by instantiate_agent for CLI/serve runs)."""
        self._policy = gate

    def set_cost_gate(self, gate: CostGate) -> None:
        """Attach a cost-budget gate (called by instantiate_agent for CLI/serve runs)."""
        self._cost = gate

    def set_capability_gate(self, gate: CapabilityGate) -> None:
        """Attach a per-skill-call capability gate (rule 11, via instantiate_agent)."""
        self._capabilities = gate

    def set_security_gate(self, gate: SecurityGateProtocol) -> None:
        """Attach the input/output security gate (rules 8 & 9, via instantiate_agent)."""
        self._security = gate

    def set_run_limits(
        self, *, max_run_tokens: int | None = None, max_turns: int | None = None
    ) -> None:
        """Set per-run limits (None = unlimited): token cap + LLM-completion (turn) cap."""
        self._max_run_tokens = max_run_tokens
        self._max_turns = max_turns

    @property
    def provider(self) -> str | None:
        return self.llm.model

    def _pre_run_gates(self, data: InputT) -> int | None:
        """Policy check then budget reservation; audit a block and re-raise if either trips.

        Returns the cost-reservation handle (or None) for the caller to settle after the run.
        Shared by run and run_stream.
        """
        try:
            self._policy.check()       # capability policy — checked FIRST (no I/O)
            return self._cost.reserve()  # budget — atomically reserve (or legacy check)
        except PolicyViolation as exc:
            self._write_block(
                data, exc, "escalated" if isinstance(exc, PolicyEscalation) else "denied"
            )
            raise
        except BudgetExceeded as exc:
            self._write_block(data, exc, "budget_exceeded")
            raise

    def run(self, data: InputT) -> OutputT:
        """Security input gate, policy + budget pre-checks, instrumented run, output gate, audit.

        The security gate (rules 8 & 9) is a no-op by default and injected on gated paths
        (`instantiate_agent(security_gate=...)` — the CLI). Its checks run OUTSIDE the
        capability `_execute` window, so the gate's own security skills stay exempt (S1)."""
        self._security.check_input(data.model_dump_json())  # rule 8: screen input first
        handle = self._pre_run_gates(data)  # policy + atomic budget reservation
        token = _audit_depth.set(_depth() + 1)
        is_root = _depth() == 1
        output: OutputT | None = None
        try:
            # Capability gate active only for THIS agent's `_execute` window, so
            # framework skills invoked outside it (e.g. the security gate) are exempt.
            cap_token = _active_capabilities.set(self._capabilities)
            try:
                output = super().run(data)
            finally:
                _active_capabilities.reset(cap_token)
            self._verify(data, output)  # agent post-condition (fail-closed) before success
            self._security.check_output(output.model_dump_json())  # rule 9: screen output
            return output
        finally:
            try:
                self._write_audit(data, output, is_root)
            finally:
                # Settle AFTER audit records the real cost: during the tiny window both the
                # reservation AND the committed cost count (over-count — the safe direction for
                # a budget gate). Cost accounting relies on audit.log() succeeding (it is the
                # ledger); if that best-effort write fails, budget tracking degrades as it does
                # for any audit-backed gate — an inherent, documented property.
                self._cost.settle(handle)
                _audit_depth.reset(token)

    def run_stream(self, data: InputT) -> Generator[str, None, None]:
        """Streaming analog of run(): same policy/cost pre-gates, instrumented stream, audit post.

        Returned as `Generator` so the 3b transport can `.close()` it to cancel. The pre-gates run
        on the first `next()`, before any delta, so a deny/over-budget raises before the first
        piece; nothing runs if the generator is never iterated. The output security gate is NOT
        here — it wraps the deltas at the serve boundary (slice 3b), like the non-streaming gate.
        """
        handle = self._pre_run_gates(data)  # policy + atomic budget reservation
        token = _audit_depth.set(_depth() + 1)
        is_root = _depth() == 1
        try:
            cap_token = _active_capabilities.set(self._capabilities)
            try:
                yield from self._instrument_stream(self._stream(data))
            finally:
                _active_capabilities.reset(cap_token)
        finally:
            try:
                # output=None: a stream has no single typed Output
                self._write_audit(data, None, is_root)
            finally:
                self._cost.settle(handle)
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

    def _enforce_token_cap(self) -> None:
        """Raise TokenCapExceeded if the active run has passed its per-run token cap.

        Called after each usage accrual, so a runaway run aborts before the NEXT LLM call."""
        cap = self._max_run_tokens
        if cap is not None and self._active_ctx is not None:
            used = self._active_ctx.input_tokens + self._active_ctx.output_tokens
            if used > cap:
                raise TokenCapExceeded(
                    f"agent {self.name!r} exceeded its per-run token cap: {used} > {cap}"
                )

    def _count_turn(self) -> None:
        """Increment the run's completion count and enforce max_turns (runaway-loop guard)."""
        if self._active_ctx is None:
            return
        self._active_ctx.turns += 1
        cap = self._max_turns
        if cap is not None and self._active_ctx.turns > cap:
            raise TurnLimitExceeded(
                f"agent {self.name!r} exceeded its max_turns: {self._active_ctx.turns} > {cap}"
            )

    def _verify(self, data: InputT, output: OutputT) -> None:
        """Post-`_execute` hook. Default no-op; override to assert output post-conditions.

        Raising fails the run (fail-closed) before the output leaves the agent — a cheap
        "check before declaring success" rail. Not invoked for streaming runs."""
        return

    def complete(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> LLMResponse:
        """Run an LLM completion, accumulating tokens/cost into the active run."""
        response = self.llm.complete(messages, model_params)
        if self._active_ctx is not None:
            self._active_ctx.add_usage(response.usage, response.cost_usd)
            self._count_turn()
            self._enforce_token_cap()
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
            self._count_turn()
            self._enforce_token_cap()

    @abstractmethod
    def _execute(self, data: InputT) -> OutputT: ...
