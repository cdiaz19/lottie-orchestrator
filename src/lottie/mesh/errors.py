"""Typed mesh errors. Transport-agnostic — no typer."""

from __future__ import annotations


class MeshError(Exception):
    """Base for all mesh-orchestration errors."""


class CapabilityViolation(MeshError):
    """The supervisor routed to a worker not declared in config.yaml `workers`."""


class MeshStepLimitExceeded(MeshError):
    """The routing loop hit `max_steps` without reaching FINISH."""
