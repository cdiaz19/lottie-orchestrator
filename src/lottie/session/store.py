"""Persist session state to `.lottie/sessions/<id>/state.json`.

Generalises the mesh durable-resume machinery (#17) from LangGraph threads to plain
`BaseAgent` runs: any agent can now write incremental progress, exit, and be resumed.

Two guards, both learned the hard way elsewhere in this codebase:

* **Session ids are validated before they touch a path.** `Path(base) / "../../etc"`
  silently escapes — the same hole PR #35 caught in the distill drafts.
* **Progress is screened on write.** It round-trips into a future run, so an agent that
  stores raw LLM output would otherwise have a way to smuggle instructions across process
  boundaries. Same reasoning as rule 13b for memory writes.
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

from lottie.security.content_gate import ContentGate, ContentRejected
from lottie.session.schema import SessionRun, SessionState


class SessionRejected(ContentRejected):
    """Session progress failed its write-time security screen. Carries no content."""


class InvalidSessionId(ValueError):
    """A session id that would escape the sessions directory, or is otherwise unusable."""


class SessionNotFound(FileNotFoundError):
    """No session exists with the requested id."""


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def safe_session_id(session_id: str) -> str:
    """Validate a session id before it is ever joined onto a path."""
    if not _ID_RE.match(session_id):
        raise InvalidSessionId(
            f"invalid session id {session_id!r}: must match {_ID_RE.pattern}"
        )
    return session_id


def session_gate() -> ContentGate:
    """The screen every session write passes. Progress round-trips into a future run."""
    return ContentGate(source="session-write", error=SessionRejected, label="session write")


class SessionStore:
    """Reads and writes session state under `<root>/.lottie/sessions/`."""

    def __init__(self, root: Path) -> None:
        self._base = root / ".lottie" / "sessions"

    def path(self, session_id: str) -> Path:
        return self._base / safe_session_id(session_id) / "state.json"

    def load(self, session_id: str) -> SessionState | None:
        """Return the stored state, or None when the session does not exist yet."""
        target = self.path(session_id)
        if not target.is_file():
            return None
        return SessionState.model_validate_json(target.read_text(encoding="utf-8"))

    def require(self, session_id: str) -> SessionState:
        """Like `load`, but raises when the session is absent."""
        state = self.load(session_id)
        if state is None:
            raise SessionNotFound(f"no session named {session_id!r}")
        return state

    def start(self, session_id: str, agent: str) -> SessionState:
        """Load an existing session, or create a fresh one. Never overwrites."""
        existing = self.load(session_id)
        if existing is not None:
            return existing
        now = time.time()
        return SessionState(
            session_id=safe_session_id(session_id),
            agent=agent,
            created_at=now,
            updated_at=now,
        )

    def save(self, state: SessionState) -> SessionState:
        """Screen and persist. Returns the state actually written, with `updated_at` set.

        Returning the stamped state (rather than the path) keeps the caller's in-memory
        copy identical to disk; returning a path invited callers to keep a stale object.
        Use `path(session_id)` when the location is what you need.

        Screened twice, deliberately. The serialised form catches structure-level payloads
        and secrets. But JSON punctuation sits between adjacent values, so
        `{"a": "ignore all previous", "b": "instructions now"}` does NOT match an injection
        pattern once serialised — the halves are separated by `", "b": "`. The second pass
        screens the values joined by newline, which is how they would read if a consumer
        ever reassembled them. Run history is excluded: it is hash-only by construction and
        re-screening it on every append would be quadratic.
        """
        gate = session_gate()
        gate.check(state.model_copy(update={"runs": []}).model_dump_json())
        gate.check("\n".join(str(v) for v in state.progress.values()))
        target = self.path(state.session_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        stamped = state.model_copy(update={"updated_at": time.time()})
        target.write_text(stamped.model_dump_json(indent=2), encoding="utf-8")
        return stamped

    def record_run(self, state: SessionState, run: SessionRun) -> SessionState:
        """Append a run to the session's history (hash-only) and return the new state."""
        return state.model_copy(update={"runs": [*state.runs, run]})

    def list(self) -> list[str]:
        """Every session id on disk, sorted."""
        if not self._base.is_dir():
            return []
        return sorted(p.name for p in self._base.iterdir() if (p / "state.json").is_file())

    def delete(self, session_id: str) -> bool:
        """Remove a session. Returns False when it did not exist."""
        target = self._base / safe_session_id(session_id)
        if not target.is_dir():
            return False
        shutil.rmtree(target)
        return True
