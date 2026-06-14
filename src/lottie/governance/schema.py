"""Typed contracts for the governance audit trail."""

from __future__ import annotations

from pydantic import BaseModel


class AuditRecord(BaseModel):
    """One immutable record of an agent run. Content is stored as sha256, never raw."""

    ts: str  # ISO-8601 UTC
    agent: str
    provider: str | None
    status: str  # "ok" | "error" | "denied" | "escalated"
    root: bool  # True = top-level run; False = nested (e.g. a mesh worker)
    input_sha256: str
    output_sha256: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    error: str | None
