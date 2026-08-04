"""Thin middleware adapters over `BaseAgent`'s existing cross-cutting calls (V3 S2).

Every adapter here does exactly what the corresponding line of `BaseAgent.run` did
before the swap-in — no logic is relocated, no behaviour changes. Only the *sequencer*
moves: `run()` stops hand-ordering these steps and lets `Pipeline` compose them.

Ownership moves later. S3 gives security/policy/cost/capability to their own subsystems,
S4 turns audit into an event subscriber, S5 moves recall/reflect/verify. Keeping S2 to
pure adapters is what makes the existing test suite a real proof: if the ordering is
faithful, ~1387 tests pass untouched; if it is not, they fail and the diff is small
enough to bisect.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

from lottie.governance.capability import _active_capabilities
from lottie.governance.cost import BudgetExceeded
from lottie.governance.policy import PolicyEscalation, PolicyViolation
from lottie.runtime.context import ExecutionContext
from lottie.runtime.middleware import Next, Order

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lottie.core.base_agent import BaseAgent


class _Agent(Protocol):
    """The slice of `BaseAgent` these adapters touch.

    A Protocol rather than the class so the adapters stay honest about their coupling:
    each one names exactly what it needs, which is also the checklist for S3-S5 when the
    concerns move out to their owning subsystems.
    """


class SecurityInputMiddleware:
    """Rule 8 — screen the input before anything else runs."""

    name = "security_input"
    order = Order.SECURITY_INPUT

    def __init__(self, agent: BaseAgent[BaseModel, BaseModel]) -> None:
        self._agent = agent

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        self._agent._security.check_input(ctx.input.model_dump_json())
        return nxt(ctx)


class PolicyMiddleware:
    """Capability policy. Audits the block itself, as `_pre_run_gates` did."""

    name = "policy"
    order = Order.POLICY

    def __init__(self, agent: BaseAgent[BaseModel, BaseModel]) -> None:
        self._agent = agent

    def _check(self, ctx: ExecutionContext) -> None:
        try:
            self._agent._policy.check()
        except PolicyViolation as exc:
            self._agent._write_block(
                ctx.input, exc, "escalated" if isinstance(exc, PolicyEscalation) else "denied"
            )
            raise

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        self._check(ctx)
        return nxt(ctx)

    @contextmanager
    def scope(self, ctx: ExecutionContext) -> Iterator[None]:
        self._check(ctx)
        yield


class CostMiddleware:
    """Atomic budget reservation, settled in a `finally` so a denied run still releases."""

    name = "cost"
    order = Order.COST

    def __init__(self, agent: BaseAgent[BaseModel, BaseModel]) -> None:
        self._agent = agent

    def _reserve(self, ctx: ExecutionContext) -> int | None:
        try:
            return self._agent._cost.reserve()
        except BudgetExceeded as exc:
            self._agent._write_block(ctx.input, exc, "budget_exceeded")
            raise

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        handle = self._reserve(ctx)
        try:
            return nxt(ctx)
        finally:
            # Settles AFTER audit (order 38 posts first): during the tiny window both the
            # reservation and the committed cost count, which over-counts — the safe
            # direction for a budget gate.
            self._agent._cost.settle(handle)

    @contextmanager
    def scope(self, ctx: ExecutionContext) -> Iterator[None]:
        """Streaming form: the reservation is held for the WHOLE stream.

        This is the reason `ScopedMiddleware` exists. Under the plain `__call__` contract
        the `finally` would fire when the generator object was created, settling the
        budget before a single delta was produced.
        """
        handle = self._reserve(ctx)
        try:
            yield
        finally:
            self._agent._cost.settle(handle)


class SessionMiddleware:
    """Post-run session bookkeeping (V2 S5b). Best-effort inside the agent method."""

    name = "session"
    order = Order.SESSION

    def __init__(self, agent: BaseAgent[BaseModel, BaseModel]) -> None:
        self._agent = agent

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        try:
            return nxt(ctx)
        finally:
            self._agent._record_session_run(ctx.input)


class TrajectoryMiddleware:
    """Post-run episodic write-back (V2 S3a). Best-effort inside the agent method."""

    name = "trajectory"
    order = Order.TRAJECTORY

    def __init__(self, agent: BaseAgent[BaseModel, BaseModel]) -> None:
        self._agent = agent

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        output: BaseModel | None = None
        try:
            output = nxt(ctx)
            return output
        finally:
            self._agent._persist_trajectory(ctx.input, output)


class AuditMiddleware:
    """One immutable record per run. Posts before trajectory, session, and settle."""

    name = "audit"
    order = Order.AUDIT

    def __init__(self, agent: BaseAgent[BaseModel, BaseModel]) -> None:
        self._agent = agent

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        output: BaseModel | None = None
        try:
            output = nxt(ctx)
            return output
        finally:
            # `is_root` is captured by DepthMiddleware on the way in, not re-read here —
            # which is precisely why moving the depth reset earlier is unobservable.
            is_root = bool(ctx.scoped("depth").get("is_root", False))
            self._agent._write_audit(ctx.input, output, is_root)

    @contextmanager
    def scope(self, ctx: ExecutionContext) -> Iterator[None]:
        """Streaming form. `output=None` — a stream has no single typed Output, which is
        exactly what `run_stream` recorded before the swap-in."""
        try:
            yield
        finally:
            is_root = bool(ctx.scoped("depth").get("is_root", False))
            self._agent._write_audit(ctx.input, None, is_root)


class DepthMiddleware:
    """Run-depth tracking for the audit `root` flag.

    Sets the ContextVar on the way in and captures `is_root` into scoped state so
    `AuditMiddleware` never has to re-read the counter.
    """

    name = "depth"
    order = Order.DEPTH

    def __init__(self, agent: BaseAgent[BaseModel, BaseModel]) -> None:
        self._agent = agent

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        with self.scope(ctx):
            return nxt(ctx)

    @contextmanager
    def scope(self, ctx: ExecutionContext) -> Iterator[None]:
        from lottie.core.base_agent import _audit_depth, _depth

        token = _audit_depth.set(_depth() + 1)
        ctx.scoped("depth")["is_root"] = _depth() == 1
        try:
            yield
        finally:
            _audit_depth.reset(token)


class RecallMiddleware:
    """Best-effort recall-as-data before `_execute`, cleared on the way out."""

    name = "recall"
    order = Order.RECALL

    def __init__(self, agent: BaseAgent[BaseModel, BaseModel]) -> None:
        self._agent = agent

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        self._agent._load_recall()
        try:
            return nxt(ctx)
        finally:
            self._agent._recall_prefix = ""


class ReflectMiddleware:
    """Opt-in post-run reflexive write-back. Only fires on a successful run."""

    name = "reflect"
    order = Order.REFLECT

    def __init__(self, agent: BaseAgent[BaseModel, BaseModel]) -> None:
        self._agent = agent

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        output = nxt(ctx)
        # Deliberately NOT in a finally: reflection distils a completed run, and a failed
        # run has no outcome to learn from. Matches `run()`'s success-path placement.
        self._agent._maybe_reflect(ctx.input, output)
        return output


class SecurityOutputMiddleware:
    """Rule 9 — screen the output before it leaves the agent."""

    name = "security_output"
    order = Order.SECURITY_OUTPUT

    def __init__(self, agent: BaseAgent[BaseModel, BaseModel]) -> None:
        self._agent = agent

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        output = nxt(ctx)
        self._agent._security.check_output(output.model_dump_json())
        return output


class VerifyMiddleware:
    """Agent post-condition hook, fail-closed before success is declared."""

    name = "verify"
    order = Order.VERIFY

    def __init__(self, agent: BaseAgent[BaseModel, BaseModel]) -> None:
        self._agent = agent

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        output = nxt(ctx)
        self._agent._verify(ctx.input, output)
        return output


class CapabilityMiddleware:
    """Rule 11 — the gate is active ONLY for the `_execute` window.

    Innermost on purpose: it must reset before `VerifyMiddleware`'s post phase, because
    `_verify` is user code that may call a skill and today runs with the gate already
    released.
    """

    name = "capability"
    order = Order.CAPABILITY

    def __init__(self, agent: BaseAgent[BaseModel, BaseModel]) -> None:
        self._agent = agent

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        with self.scope(ctx):
            return nxt(ctx)

    @contextmanager
    def scope(self, ctx: ExecutionContext) -> Iterator[None]:
        token = _active_capabilities.set(self._agent._capabilities)
        try:
            yield
        finally:
            _active_capabilities.reset(token)


#: The standard chain, in declaration order. `Pipeline` sorts by `order`, so this list is
#: for readability only — the authority is `runtime.middleware.Order`.
STANDARD_CHAIN = (
    SecurityInputMiddleware,
    PolicyMiddleware,
    CostMiddleware,
    SessionMiddleware,
    TrajectoryMiddleware,
    AuditMiddleware,
    DepthMiddleware,
    RecallMiddleware,
    ReflectMiddleware,
    SecurityOutputMiddleware,
    VerifyMiddleware,
    CapabilityMiddleware,
)
