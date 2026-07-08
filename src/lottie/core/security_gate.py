"""Core-side security-gate seam (rules 8 & 9) for the BaseAgent chokepoint.

`BaseAgent.run` screens input then output through an injected gate. The concrete gate
(`serve.security.SecurityGate`) runs the security skills and raises the serve-layer
violation types; it cannot be imported here (it would cycle core -> serve). So core
defines only a structural Protocol + a no-op default, and the real gate is injected by a
caller that may import serve (the CLI). BaseAgent never references the violation types —
it calls the gate and lets whatever it raises propagate.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SecurityGateProtocol(Protocol):
    """Input/output screen. Implementations raise (fail-closed) on a tripped check."""

    def check_input(self, text: str) -> None: ...

    def check_output(self, text: str) -> None: ...


class NullSecurityGate:
    """No-op gate — the BaseAgent default (ungated). Direct construction stays unenforced;
    a real gate is attached by `instantiate_agent(security_gate=...)` on gated paths."""

    def check_input(self, text: str) -> None:
        return

    def check_output(self, text: str) -> None:
        return
