"""Session middleware — post-run session bookkeeping as a mounted module (V3 S5).

Owned by the session subsystem. Takes the store and a usage reader; no `BaseAgent`.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import BaseModel

from lottie.runtime.context import ExecutionContext
from lottie.runtime.middleware import Next, Order
from lottie.session.schema import SessionRun, SessionState
from lottie.session.store import SessionStore


class SessionMiddleware:
    """Append this run to the session's history, hash-only.

    Best-effort: a store failure must never fail the run. The history records THAT the
    session progressed and what it cost, never the content — the same discipline as the
    audit ledger.
    """

    name = "session"
    order = Order.SESSION

    def __init__(
        self,
        store: SessionStore | None,
        get_state: Callable[[], SessionState | None],
        set_state: Callable[[SessionState], None],
        usage: Callable[[], SessionRun | None],
        hasher: Callable[[BaseModel], str],
    ) -> None:
        self._store = store
        self._get_state = get_state
        self._set_state = set_state
        self._usage = usage
        self._hasher = hasher

    def _record(self, ctx: ExecutionContext) -> None:
        state = self._get_state()
        if state is None or self._store is None:
            return
        partial = self._usage()
        if partial is None:
            return
        try:
            recorded = self._store.record_run(
                state,
                partial.model_copy(
                    update={
                        "ts": datetime.now(UTC).timestamp(),
                        "input_sha256": self._hasher(ctx.input),
                    }
                ),
            )
            self._set_state(self._store.save(recorded))
        except Exception as exc:  # best-effort — never fail the run
            warnings.warn(f"session record failed: {exc}", stacklevel=2)

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        try:
            return nxt(ctx)
        finally:
            self._record(ctx)
