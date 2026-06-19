"""Serve-layer error hierarchy. Dependency-free so any serve module can raise these
without import cycles (e.g. security.py raising SecurityViolation)."""

from __future__ import annotations


class ServeError(Exception):
    """Base for all serving-core errors. Transports map subclasses to status codes."""


class SecurityViolation(ServeError):
    """Raised by the SecurityGate when an input/output check fails (fail-closed)."""


class InputSecurityViolation(SecurityViolation):
    """The input gate (sanitize / injection) rejected the request content."""


class OutputSecurityViolation(SecurityViolation):
    """The output gate (validate / secret) withheld the produced content.

    Carries the run's token counts so an HTTP transport can report `usage` on the
    withheld response (the agent already ran). Defaults to zero for callers that
    raise it without metrics (e.g. the gate itself).
    """

    def __init__(
        self, message: str, *, input_tokens: int = 0, output_tokens: int = 0
    ) -> None:
        super().__init__(message)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class NotResumable(ServeError):
    """The agent exists but cannot be resumed (not a mesh / no HITL)."""


class ThreadNotFound(ServeError):
    """No checkpoint exists for the given thread_id (never existed or pruned)."""
