"""Security middleware — rules 8 and 9 as mounted modules (V3 S3).

Owned by the security subsystem rather than by `core`. These depend only on the runtime
kernel and on a `check_input`/`check_output` gate passed in; they know nothing about
`BaseAgent`, which is what lets the `core -> security` edge disappear once the module
orchestrator (S6) does the wiring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

from lottie.runtime.context import ExecutionContext
from lottie.runtime.middleware import Next, Order

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


class InputOutputGate(Protocol):
    """The slice of a security gate these modules use.

    Structural, and declared here rather than imported from `core.security_gate`: naming
    what it needs is exactly what keeps this module free of a core dependency.
    """

    def check_input(self, payload: str) -> None: ...

    def check_output(self, payload: str) -> None: ...


class SecurityInputMiddleware:
    """Rule 8 — screen the input before anything else runs."""

    name = "security_input"
    order = Order.SECURITY_INPUT

    def __init__(self, gate: InputOutputGate) -> None:
        self._gate = gate

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        self._gate.check_input(ctx.input.model_dump_json())
        return nxt(ctx)


class SecurityOutputMiddleware:
    """Rule 9 — screen the output before it leaves the agent.

    No `scope` form: it needs the output VALUE, which a context manager's `__exit__`
    never receives. Correctly absent from a streaming chain, where the gate wraps the
    deltas at the serve boundary instead.
    """

    name = "security_output"
    order = Order.SECURITY_OUTPUT

    def __init__(self, gate: InputOutputGate) -> None:
        self._gate = gate

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        output = nxt(ctx)
        self._gate.check_output(output.model_dump_json())
        return output
