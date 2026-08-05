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
from typing import TYPE_CHECKING

from pydantic import BaseModel

from lottie.governance.middleware import (
    CapabilityMiddleware,
    CostMiddleware,
    PolicyMiddleware,
)
from lottie.runtime.context import ExecutionContext
from lottie.runtime.middleware import Middleware, Next, Order
from lottie.security.middleware import (
    SecurityInputMiddleware,
    SecurityOutputMiddleware,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lottie.core.base_agent import BaseAgent





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
        # A first-class run property rather than scoped state: observers need it, and
        # reaching into another module's private slice to get it would be worse.
        ctx.root = _depth() == 1
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



def build_chain(agent: BaseAgent[BaseModel, BaseModel]) -> list[Middleware]:
    """The standard chain for an agent.

    Security, policy, cost and capability come from their OWNING subsystems (V3 S3) and
    are constructed from the gate alone. Audit left the chain entirely in S4 — it is an
    observer, so it is an `EventBus` subscriber now. The remainder are still agent-coupled
    adapters and move out in S5 (recall / reflect / verify).

    `Pipeline` sorts by `order`, so this order is for readability only — the authority is
    `runtime.middleware.Order`.
    """
    return [
        SecurityInputMiddleware(agent._security),
        PolicyMiddleware(agent._policy, agent._write_block),
        CostMiddleware(agent._cost, agent._write_block),
        SessionMiddleware(agent),
        TrajectoryMiddleware(agent),
        DepthMiddleware(agent),
        RecallMiddleware(agent),
        ReflectMiddleware(agent),
        SecurityOutputMiddleware(agent._security),
        VerifyMiddleware(agent),
        CapabilityMiddleware(agent._capabilities),
    ]
