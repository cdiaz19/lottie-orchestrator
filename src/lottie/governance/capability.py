"""Capability gate (CLAUDE.md rule 11): an agent may only call skills declared in
its config `capabilities` list. Undeclared skill calls are blocked, fail-closed.

Lives in ``governance`` (not ``security``) to stay core-free: ``core.base_agent`` and
``core.base_skill`` both import the gate + the ``_active_capabilities`` ContextVar, and
``security`` imports ``core`` (its skills extend ``BaseSkill``), so a gate in ``security``
would create a ``core -> security -> core`` cycle. Mirrors the sibling ``policy``/``cost``
gates, which ``base_agent`` already imports acyclically.

Imports only stdlib -- no core/project/serve deps.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterable


class CapabilityDenied(Exception):
    """A skill was invoked that the calling agent did not declare (fail-closed).

    Governance-local (like ``PolicyViolation``/``BudgetExceeded``); NOT a subclass of
    ``serve.errors.SecurityViolation``, which would cycle back through ``serve``.
    """


class CapabilityGate:
    """Checks a skill's capability name against an agent's declared whitelist."""

    def __init__(self, capabilities: Iterable[str]) -> None:
        self._allowed = frozenset(capabilities)

    def check(self, capability: str) -> None:
        """Raise ``CapabilityDenied`` if ``capability`` is not declared; else return."""
        if capability not in self._allowed:
            raise CapabilityDenied(
                f"skill {capability!r} is not a declared capability "
                f"{sorted(self._allowed)}"
            )


class NullCapabilityGate(CapabilityGate):
    """No-op gate -- the BaseAgent default and the 'no capabilities declared' result."""

    def __init__(self) -> None:
        super().__init__([])

    def check(self, capability: str) -> None:
        return


def build_capability_gate(*, capabilities: list[str]) -> CapabilityGate:
    """Whitelist-when-nonempty: empty/absent -> NullCapabilityGate (no enforcement);
    non-empty -> strict CapabilityGate over the declared names."""
    if not capabilities:
        return NullCapabilityGate()
    return CapabilityGate(capabilities)


_NULL_GATE = NullCapabilityGate()

# Active gate for the current agent's `_execute` window. BaseAgent.run sets this to its
# own gate tightly around super().run() and resets it after; BaseSkill.run reads it via
# `active_capability_gate()`. Default None (outside any run) -> the no-op gate, so direct
# skill construction / unit tests are unenforced.
_active_capabilities: contextvars.ContextVar[CapabilityGate | None] = (
    contextvars.ContextVar("lottie_active_capabilities", default=None)
)


def active_capability_gate() -> CapabilityGate:
    """The capability gate in force for the current `_execute` window, or a no-op gate."""
    gate = _active_capabilities.get()
    return gate if gate is not None else _NULL_GATE
