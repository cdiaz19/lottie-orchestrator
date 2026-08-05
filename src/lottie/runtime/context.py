"""Per-run carrier threaded through the middleware chain.

Deliberately independent of `lottie.core`: `core/__init__.py` eagerly imports
`base_agent`, so once `BaseAgent` mounts the kernel (S2) a `lottie.core` import here
would be a circular import at package-init time. The kernel therefore depends on a
structural Protocol and a mirrored Literal instead, both pinned by `test_context.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

RunKind = Literal["agent", "skill"]
"""Mirrors `lottie.core.metrics.Kind`. Duplicated, not imported — see module docstring."""


@runtime_checkable
class UsageAccumulator(Protocol):
    """Structural view of `lottie.core.metrics.RunContext`.

    The kernel only carries and reads usage, never constructs it, so it depends on this
    Protocol rather than on the concrete dataclass.
    """

    input_tokens: int
    output_tokens: int
    cost_usd: float
    turns: int


@dataclass
class NullUsage:
    """Zeroed accumulator used when no real one is injected (kernel unit tests)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    turns: int = 0


@dataclass
class ExecutionContext:
    """Mutable per-run carrier. One instance per `Pipeline.execute` call."""

    runnable: str
    kind: RunKind
    input: BaseModel
    run_id: str
    usage: UsageAccumulator = field(default_factory=NullUsage)
    #: True when this is a TOP-LEVEL run rather than a nested mesh worker. A first-class
    #: run property, not a module detail: observers need it and would otherwise have to
    #: reach into another module's scoped state to get it.
    root: bool = False
    #: Model id for observability. None for non-LLM runnables.
    provider: str | None = None
    state: dict[str, object] = field(default_factory=dict)

    def scoped(self, module: str) -> dict[str, object]:
        """Return `module`'s private slice of `state`, creating it on first use.

        Middleware never touch `state` directly. Namespacing by module name is what stops
        two independently-authored modules from silently colliding on a key.
        """
        slot = self.state.setdefault(module, {})
        if not isinstance(slot, dict):
            raise TypeError(f"ExecutionContext.state[{module!r}] is not a dict: {type(slot)}")
        return slot
