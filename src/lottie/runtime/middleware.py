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

        pre :  check_input -> policy -> reserve -> depth set -> recall load -> cap set
        core:  [run + emit RunStarted/RunCompleted]
        post:  cap reset -> verify -> check_output -> reflect -> recall clear
               -> depth reset -> audit -> trajectory -> session -> cost settle

    which reproduces `core/base_agent.py`'s sequence as of v2.0.0, with two deliberate
    and provably unobservable deviations (see below).

    Why these values
    ----------------
    * `CAPABILITY` is the INNERMOST (90), not 50. Today `_active_capabilities` is reset
      immediately after `_execute`, BEFORE `_verify`. `_verify` is user code that may call
      a skill, so leaving the gate active there would change rule-11 enforcement. Post
      phases unroll in reverse, so the capability middleware must be innermost to reset
      first.
    * `AUDIT` (38) sits above `TRAJECTORY` (36) above `SESSION` (34) above `COST` (30), so
      the post phase unrolls audit -> trajectory -> session -> settle. Audit-before-settle
      is the load-bearing invariant documented in `BaseAgent.run`.
    * `DEPTH` (40) must be greater than `COST` (30) in the PRE phase: if the depth counter
      were incremented before the budget gate ran, a denied top-level run would be audited
      `root=False` by `_write_block`, which reads `_depth() == 0`.

    Two accepted deviations
    -----------------------
    `run()`'s current interleaving cannot be expressed as a pure onion with one middleware
    per concern: the pre phase needs COST < DEPTH, while the post phase would need
    DEPTH < COST for the depth reset to follow the cost settle. Both are unobservable and
    are accepted rather than splitting Depth into two modules:

    1. `depth set` moves before `recall load` — `_load_recall` never reads the depth.
    2. `depth reset` moves before `audit` — `CostGate.settle` never reads the depth, and
       `_write_audit` receives `is_root` as a captured parameter rather than re-reading it.

    Third-party modules (E7) pick values between these; the registry rejects collisions
    at registration.
    """

    SECURITY_INPUT = 10
    POLICY = 20
    COST = 30
    SESSION = 34
    TRAJECTORY = 36
    AUDIT = 38
    DEPTH = 40
    RECALL = 60
    REFLECT = 70
    SECURITY_OUTPUT = 75
    VERIFY = 80
    CAPABILITY = 90
