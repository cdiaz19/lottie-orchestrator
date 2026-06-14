from __future__ import annotations

from lottie.serve import SecurityViolation, ServeError
from lottie.serve.service import AgentNotFoundError


def test_security_violation_is_serve_error() -> None:
    assert issubclass(SecurityViolation, ServeError)
    assert issubclass(AgentNotFoundError, ServeError)  # subclasses still wired to the moved base


def test_security_violation_message_roundtrips() -> None:
    err = SecurityViolation("input rejected: prompt-injection detected")
    assert "prompt-injection" in str(err)
