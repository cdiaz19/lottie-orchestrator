"""Serving-core schemas — transport-agnostic agent metadata and run output."""

from __future__ import annotations

from pydantic import BaseModel


class AgentInfo(BaseModel):
    """One discovered agent, import-free (name + configured provider)."""

    name: str
    provider: str | None = None


class RunResult(BaseModel):
    """Result of one agent run: output payload plus per-run metrics."""

    agent: str
    output: dict[str, object]  # output.model_dump(); dict (not Any) keeps mypy honest
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    status: str = "complete"
    thread_id: str | None = None
    pending: dict[str, object] | None = None
