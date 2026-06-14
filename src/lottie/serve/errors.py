"""Serve-layer error hierarchy. Dependency-free so any serve module can raise these
without import cycles (e.g. security.py raising SecurityViolation)."""

from __future__ import annotations


class ServeError(Exception):
    """Base for all serving-core errors. Transports map subclasses to status codes."""


class SecurityViolation(ServeError):
    """Raised by the SecurityGate when an input/output check fails (fail-closed)."""
