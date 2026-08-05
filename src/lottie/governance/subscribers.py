"""Governance observers as event subscribers (V3 S4).

Auditing a completed run is pure observation: it reads what happened and writes a record.
Making it a `Subscriber` rather than a middleware means "best-effort" stops being a
`try/except` the author has to remember and becomes a property of the bus — `EventBus.emit`
wraps every dispatch, so a broken observer cannot fail a run and cannot starve the next
observer.

Blocked runs are NOT here. A refused run carries a governance-specific status
(`denied` / `escalated` / `budget_exceeded`) that only the gate knows, so the gates audit
their own blocks through the `BlockAudit` callback injected in S3. The kernel would
otherwise have to learn governance vocabulary to put it on an event.
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime

from lottie.governance.audit import AuditLogger
from lottie.governance.schema import AuditRecord
from lottie.runtime.events import RunCompleted, RunEvent, RunFailed


class AuditSubscriber:
    """Writes one immutable ledger record per completed or failed run.

    Everything it needs arrives on the event — hashes, usage, timing, the root flag —
    so it never reaches back into the agent. That is what lets `core` stop owning the
    audit call at all.
    """

    name = "audit"

    def __init__(self, logger: AuditLogger) -> None:
        self._logger = logger

    def on_event(self, event: RunEvent) -> None:
        if isinstance(event, RunCompleted):
            self._log(event, status="ok", error=None, output_sha256=event.output_sha256)
        elif isinstance(event, RunFailed):
            self._log(event, status="error", error=event.error, output_sha256=None)

    def _log(
        self,
        event: RunCompleted | RunFailed,
        *,
        status: str,
        error: str | None,
        output_sha256: str | None,
    ) -> None:
        # `EventBus.emit` already isolates a raising subscriber, but a record that cannot
        # even be BUILT should not surface as a bus warning about the subscriber — the
        # cause is the data, and saying so is more useful.
        try:
            record = AuditRecord(
                # The event fires from the innermost frame, microseconds after the run
                # ended, so "now" and the metrics timestamp are the same instant.
                ts=datetime.now(UTC).isoformat(),
                agent=event.runnable,
                provider=event.provider,
                status=status,
                root=event.root,
                input_sha256=event.input_sha256,
                output_sha256=output_sha256,
                input_tokens=event.input_tokens,
                output_tokens=event.output_tokens,
                cost_usd=event.cost_usd,
                latency_ms=event.latency_ms,
                error=error,
            )
        except Exception as exc:
            warnings.warn(f"audit record could not be built: {exc}", stacklevel=2)
            return
        self._logger.log(record)
