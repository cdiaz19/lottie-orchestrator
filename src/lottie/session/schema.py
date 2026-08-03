"""Typed models for session persistence (V2 S5b).

A session is how a task survives beyond one process: an agent writes incremental progress,
the process exits, and a later `lottie run --session <id>` picks up where it left off.
"""

from __future__ import annotations

from pydantic import BaseModel


class SessionRun(BaseModel):
    """One run recorded against a session.

    Hash-only, like the audit ledger: the history is here so an operator can see *that*
    the session progressed and what it cost, not to replay content.
    """

    ts: float
    status: str  # ok | error
    input_sha256: str | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None


class SessionState(BaseModel):
    """Everything a session carries across processes.

    `progress` is the agent's own structured state — whatever it needs to resume. It is
    DATA on the way back in, never instructions: an agent that reads progress and treats
    it as a directive re-opens the poisoning hole the memory subsystem closes, since a
    previous run's LLM output can reach it.
    """

    session_id: str
    agent: str
    created_at: float
    updated_at: float
    progress: dict[str, object] = {}
    runs: list[SessionRun] = []
