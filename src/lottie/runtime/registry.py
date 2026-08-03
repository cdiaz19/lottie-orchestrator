"""Module Orchestrator support — the registry S6 wires `instantiate_agent` onto.

A module declares a factory; the registry composes factories into a chain. Adding a
cross-cutting concern becomes "register one factory" instead of the four-file edit
that `BaseAgent.__init__` + `run` + `instantiate_agent` + `AgentConfig` requires today.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from lottie.runtime.events import EventBus, Subscriber
from lottie.runtime.middleware import Middleware


class ModuleConflictError(RuntimeError):
    """Two modules claim the same name or the same chain position.

    Raised at registration rather than at run time so the failure surfaces at startup,
    and so a plugin can never silently displace a security middleware.
    """


@dataclass(frozen=True)
class Deps:
    """Constructor dependencies handed to a module factory.

    Deliberately minimal. S1 has no real modules, so adding fields now would be
    inventing API ahead of need; S6 extends this once the migrated modules declare what
    they actually require. Dependencies are injected here rather than imported by the
    module, which is what keeps the subsystem-to-runtime edge one-directional.
    """

    bus: EventBus
    root: Path


type Mountable = Middleware | Subscriber


class ModuleFactory[CfgT](Protocol):
    """Builds a module from configuration, or declines to.

    Generic over the config type so the kernel never imports `AgentConfig` from
    `lottie.project` — S6 instantiates this as `ModuleFactory[AgentConfig]`.
    """

    name: str
    order: int

    def build(self, cfg: CfgT, deps: Deps) -> Mountable | None: ...


class ModuleRegistry[CfgT]:
    """Ordered collection of module factories."""

    def __init__(self) -> None:
        self._factories: list[ModuleFactory[CfgT]] = []

    def names(self) -> list[str]:
        """Registered module names, in registration order."""
        return [f.name for f in self._factories]

    def register(self, factory: ModuleFactory[CfgT]) -> None:
        """Add `factory`, rejecting name and order collisions.

        Order uniqueness is enforced across all factories, including ones that produce
        subscribers. Subscribers do not strictly need a distinct order, but a global
        rule is trivially satisfiable and leaves no ambiguity about who owns a slot.
        """
        for existing in self._factories:
            if existing.name == factory.name:
                raise ModuleConflictError(f"module name {factory.name!r} is already registered")
            if existing.order == factory.order:
                raise ModuleConflictError(
                    f"module {factory.name!r} claims order {factory.order}, "
                    f"already held by {existing.name!r}"
                )
        self._factories.append(factory)

    def build(self, cfg: CfgT, deps: Deps) -> tuple[list[Middleware], list[Subscriber]]:
        """Instantiate every enabled module, split into chain and observers.

        A factory returning None is disabled by configuration and costs nothing.
        """
        middleware: list[Middleware] = []
        subscribers: list[Subscriber] = []
        for factory in self._factories:
            mounted = factory.build(cfg, deps)
            if mounted is None:
                continue
            if isinstance(mounted, Middleware):
                middleware.append(mounted)
            else:
                subscribers.append(mounted)
        middleware.sort(key=lambda m: m.order)
        return middleware, subscribers
