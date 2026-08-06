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
from lottie.runtime.registry import ModuleConflictError
from lottie.security.middleware import (
    SecurityInputMiddleware,
    SecurityOutputMiddleware,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lottie.core.base_agent import BaseAgent







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



def build_chain(
    agent: BaseAgent[BaseModel, BaseModel], disabled: frozenset[str] = frozenset()
) -> list[Middleware]:
    """The standard chain for an agent.

    Security, policy, cost and capability come from their OWNING subsystems (V3 S3) and
    are constructed from the gate alone. Audit left the chain entirely in S4 — it is an
    observer, so it is an `EventBus` subscriber now. The remainder are still agent-coupled
    adapters. S5 moved recall, trajectory and session to their owning subsystems too;
    what remains agent-coupled is depth (a core concern), verify (a BaseAgent extension
    point) and reflect (which re-enters the agent's own `complete()` with primed budget
    state, and becomes a real module once E4 owns message assembly).

    `Pipeline` sorts by `order`, so this order is for readability only — the authority is
    `runtime.middleware.Order`.

    `disabled` drops modules by name (V3 S6, the `modules:` config block). A dropped
    module is never constructed, so it costs nothing at run time — the same "return None
    from the factory" semantics the registry uses.
    """
    modules: list[Middleware] = [
        SecurityInputMiddleware(agent._security),
        PolicyMiddleware(agent._policy, agent._write_block),
        CostMiddleware(agent._cost, agent._write_block),
        agent._session_module(),
        agent._trajectory_module(),
        DepthMiddleware(agent),
        agent._recall_module(),
        agent._reflect_module(),
        SecurityOutputMiddleware(agent._security),
        VerifyMiddleware(agent),
        CapabilityMiddleware(agent._capabilities),
    ]
    if disabled:
        modules = [m for m in modules if m.name not in disabled]
    _reject_order_conflicts(modules)
    return modules


def _reject_order_conflicts(modules: list[Middleware]) -> None:
    """Fail at composition, not at run time.

    Two modules claiming one chain position is ambiguous about who owns that slot, and a
    plugin (E7) must never be able to silently displace a security gate. Raising here
    surfaces it at startup where an operator sees it.
    """
    seen: dict[int, str] = {}
    for module in modules:
        clash = seen.get(module.order)
        if clash is not None:
            raise ModuleConflictError(
                f"module {module.name!r} claims order {module.order}, "
                f"already held by {clash!r}"
            )
        seen[module.order] = module.name


#: Every module name the standard chain can mount. `lottie modules` and `lottie doctor`
#: use it to reject a `modules:` block naming something that does not exist — a silent
#: typo there would leave a security gate mounted when the operator believed otherwise.
KNOWN_MODULES = (
    "security_input",
    "policy",
    "cost",
    "session",
    "trajectory",
    "depth",
    "recall",
    "reflect",
    "security_output",
    "verify",
    "capability",
)
