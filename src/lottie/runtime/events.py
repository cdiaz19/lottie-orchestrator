"""Event Runtime — the fail-open observation stream the pipeline emits onto.

Two rules, both structural rather than conventional:

1. **A subscriber can never break a run.** `EventBus.emit` wraps every dispatch; an
   exception becomes a warning and the next subscriber still runs. This replaces the
   hand-rolled try/except that best-effort observers repeat today in `_write_audit`,
   `_persist_trajectory`, and `_record_session_run`.
2. **Events carry scalars and hashes only, never raw content.** The bus is an
   observation surface that the V3 Plugin SDK (E7) opens to third parties, so a raw
   payload here would be an exfiltration channel. `test_events.py` enforces it against
   every event model — a subscriber that genuinely needs content must be a middleware,
   where trust is explicit.
"""

from __future__ import annotations

import warnings
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from lottie.runtime.context import RunKind


class RunEvent(BaseModel):
    """Base for every lifecycle event.

    Frozen so one subscriber cannot mutate what later subscribers observe.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    runnable: str
    kind: RunKind


class RunStarted(RunEvent):
    """Emitted from the innermost frame, immediately before the real work runs."""

    input_sha256: str


class RunCompleted(RunEvent):
    """Emitted from the innermost frame after a successful run, before any post-phase."""

    input_sha256: str
    output_sha256: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float


class RunFailed(RunEvent):
    """The core execution raised."""

    input_sha256: str
    error: str
    latency_ms: float


class RunBlocked(RunEvent):
    """A middleware aborted the run before the core frame was ever entered.

    `blocked_by` is the name of the last middleware entered — the one that refused.
    """

    input_sha256: str
    blocked_by: str
    error: str


class Subscriber(Protocol):
    """A fail-open observer. Raising is tolerated and warned, never propagated."""

    name: str

    def on_event(self, event: RunEvent) -> None: ...


class EventBus:
    """Synchronous, in-process, registration-ordered fan-out."""

    def __init__(self) -> None:
        self._subs: list[Subscriber] = []

    def subscribe(self, sub: Subscriber) -> None:
        self._subs.append(sub)

    def emit(self, event: RunEvent) -> None:
        """Dispatch to every subscriber, isolating each failure.

        Deliberately swallows: an observer must never be able to fail a run, and one
        broken observer must never starve the ones registered after it.
        """
        for sub in self._subs:
            try:
                sub.on_event(event)
            except Exception as exc:
                warnings.warn(
                    f"subscriber {sub.name!r} failed on {type(event).__name__}: {exc}",
                    stacklevel=2,
                )
