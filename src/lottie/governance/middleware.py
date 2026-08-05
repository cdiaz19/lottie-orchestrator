"""Governance middleware — policy, budget, and capability as mounted modules (V3 S3).

Owned by the governance subsystem. Each depends only on the runtime kernel and on its own
gate; none knows about `BaseAgent`. Policy and Cost audit their own blocks through an
injected callback rather than reaching back into the agent, which is what removes the
`core -> governance` edge once S6 does the wiring.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Literal

from pydantic import BaseModel

from lottie.governance.capability import CapabilityGate, _active_capabilities
from lottie.governance.cost import BudgetExceeded, CostGate
from lottie.governance.policy import PolicyEscalation, PolicyGate, PolicyViolation
from lottie.runtime.context import ExecutionContext
from lottie.runtime.middleware import Next, Order

BlockStatus = Literal["denied", "escalated", "budget_exceeded"]

#: Records a refused run in the audit ledger. Injected so these modules never import the
#: audit logger — the blocked-run record is the caller's concern, not the gate's.
BlockAudit = Callable[[BaseModel, Exception, BlockStatus], None]


class PolicyMiddleware:
    """Declarative capability policy, checked before any I/O."""

    name = "policy"
    order = Order.POLICY

    def __init__(self, gate: PolicyGate, audit_block: BlockAudit) -> None:
        self._gate = gate
        self._audit_block = audit_block

    def _check(self, ctx: ExecutionContext) -> None:
        try:
            self._gate.check()
        except PolicyViolation as exc:
            status: BlockStatus = "escalated" if isinstance(exc, PolicyEscalation) else "denied"
            self._audit_block(ctx.input, exc, status)
            raise

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        self._check(ctx)
        return nxt(ctx)

    @contextmanager
    def scope(self, ctx: ExecutionContext) -> Iterator[None]:
        self._check(ctx)
        yield


class CostMiddleware:
    """Atomic budget reservation, settled on the way out."""

    name = "cost"
    order = Order.COST

    def __init__(self, gate: CostGate, audit_block: BlockAudit) -> None:
        self._gate = gate
        self._audit_block = audit_block

    def _reserve(self, ctx: ExecutionContext) -> int | None:
        try:
            return self._gate.reserve()
        except BudgetExceeded as exc:
            self._audit_block(ctx.input, exc, "budget_exceeded")
            raise

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        handle = self._reserve(ctx)
        try:
            return nxt(ctx)
        finally:
            # Settles AFTER audit (order 38 posts first): during that window both the
            # reservation and the committed cost count, which over-counts — the safe
            # direction for a budget gate.
            self._gate.settle(handle)

    @contextmanager
    def scope(self, ctx: ExecutionContext) -> Iterator[None]:
        """Streaming form: the reservation is held for the WHOLE stream.

        Under the plain `__call__` contract the `finally` would fire when the generator
        object was created, settling the budget before a single delta was produced.
        """
        handle = self._reserve(ctx)
        try:
            yield
        finally:
            self._gate.settle(handle)


class CapabilityMiddleware:
    """Rule 11 — the gate is active ONLY for the `_execute` window.

    Innermost on purpose: it must reset before `VerifyMiddleware`'s post phase, because
    `_verify` is user code that may call a skill and today runs with the gate released.
    """

    name = "capability"
    order = Order.CAPABILITY

    def __init__(self, gate: CapabilityGate) -> None:
        self._gate = gate

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        with self.scope(ctx):
            return nxt(ctx)

    @contextmanager
    def scope(self, ctx: ExecutionContext) -> Iterator[None]:
        token = _active_capabilities.set(self._gate)
        try:
            yield
        finally:
            _active_capabilities.reset(token)
