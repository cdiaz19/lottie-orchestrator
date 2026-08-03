"""Lifecycle Hooks — the ordered, abort-capable wrapper contract.

A middleware receives the context and a `nxt` callable. Calling `nxt(ctx)` runs the rest
of the chain (ultimately the real work); NOT calling it aborts the run. Code before
`nxt` is the pre-phase, code after is the post-phase, and a `try/finally` around `nxt`
is how a concern guarantees cleanup — the property a pure event bus cannot express,
and the reason gates are middleware rather than subscribers.

Post-phases run in REVERSE order of pre-phases (onion nesting). The `Order` values below
are chosen so that unrolling reproduces `BaseAgent.run`'s existing sequence exactly.
See V3 spec section 4.5; `test_middleware.py` pins the relationships that matter.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from lottie.runtime.context import ExecutionContext

type Next = Callable[[ExecutionContext], BaseModel]
"""Continuation into the rest of the chain.

Typed as `BaseModel` rather than `Any`: every runnable output is a pydantic model
(rule 2), so `BaseModel` is the true bound. `Pipeline` narrows it back to the concrete
`OutputT` with a single `cast` at its public boundary, so the kernel carries no `Any`.
"""


@runtime_checkable
class Middleware(Protocol):
    """One lifecycle hook. Lower `order` runs earlier in the pre-phase.

    Structural — a middleware never inherits from this. `runtime_checkable` so the
    registry can tell a middleware from a subscriber: the discriminator is `order`,
    which a subscriber does not have.
    """

    name: str
    order: int

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel: ...


class Order:
    """Canonical chain positions.

    Unrolled, the standard chain produces:

        check_input -> policy -> reserve -> depth -> capability -> recall
          -> [core frame: run + emit RunCompleted]
          -> verify -> check_output -> reflect -> recall clear
          -> capability reset -> depth reset
          -> trajectory -> session -> cost settle

    which is `core/base_agent.py:427-467` as of v2.0.0.

    TRAJECTORY and SESSION sit between COST and DEPTH because their work happens in the
    POST phase, which unrolls in reverse: a higher order means an earlier post-phase.
    Both must land after the depth reset and before the cost settle, and Trajectory must
    be outside Session to keep them in their current relative order.

    Third-party modules (E7) pick values between these; the registry rejects collisions
    at registration.
    """

    SECURITY_INPUT = 10
    POLICY = 20
    COST = 30
    SESSION = 34
    TRAJECTORY = 36
    DEPTH = 40
    CAPABILITY = 50
    RECALL = 60
    REFLECT = 70
    SECURITY_OUTPUT = 75
    VERIFY = 80
