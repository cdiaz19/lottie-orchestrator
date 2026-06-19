"""Pure dict builders for the Lottie-native REST surface. No Starlette import — unit-testable bare.

The REST `run` response is a serialized RunResult; the withhold response is the same shape with the
output stripped and status="withheld" (the run executed; the body is withheld, not an error)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from lottie.serve.schema import AgentInfo, RunResult


class ResumeDecision(BaseModel):
    """A human HITL decision, mesh-import-free (converted to ApprovalDecision in the service)."""

    action: Literal["approve", "reject"]
    edited_input: dict[str, str] = {}


class ResumeRequest(BaseModel):
    thread_id: str
    decision: ResumeDecision


def agent_list_dict(infos: list[AgentInfo]) -> dict[str, object]:
    """`GET /v1/agents` body — every agent's name + provider."""
    return {"agents": [{"name": i.name, "provider": i.provider} for i in infos]}


def agent_detail_dict(
    name: str, provider: str | None, input_schema: dict[str, object]
) -> dict[str, object]:
    """`GET /v1/agents/{name}` body — provider + the Input JSON schema."""
    return {"name": name, "provider": provider, "input_schema": input_schema}


def run_result_dict(result: RunResult) -> dict[str, object]:
    """`POST /v1/agents/{name}/run` success body — the serialized RunResult."""
    return {
        "agent": result.agent,
        "output": result.output,
        "status": result.status,
        "latency_ms": result.latency_ms,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": result.cost_usd,
        "thread_id": result.thread_id,
        "pending": result.pending,
    }


def withheld_dict(
    agent: str, *, input_tokens: int, output_tokens: int
) -> dict[str, object]:
    """Output-withhold body — RunResult shape with the output stripped, status=withheld.

    Usage is still reported (tokens were spent); the withheld output is never included."""
    return {
        "agent": agent,
        "output": {},
        "status": "withheld",
        "latency_ms": 0.0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": 0.0,
        "thread_id": None,
        "pending": None,
    }
