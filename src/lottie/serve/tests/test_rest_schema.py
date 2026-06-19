from __future__ import annotations

from lottie.serve.rest_schema import (
    agent_detail_dict,
    agent_list_dict,
    run_result_dict,
    withheld_dict,
)
from lottie.serve.schema import AgentInfo, RunResult


def test_agent_list_dict() -> None:
    infos = [AgentInfo(name="digest", provider="anthropic/x"), AgentInfo(name="m", provider=None)]
    assert agent_list_dict(infos) == {
        "agents": [
            {"name": "digest", "provider": "anthropic/x"},
            {"name": "m", "provider": None},
        ]
    }


def test_agent_detail_dict() -> None:
    schema: dict[str, object] = {"type": "object", "properties": {"query": {"type": "string"}}}
    assert agent_detail_dict("digest", "anthropic/x", schema) == {
        "name": "digest",
        "provider": "anthropic/x",
        "input_schema": schema,
    }


def test_run_result_dict() -> None:
    r = RunResult(
        agent="digest", output={"result": "ok"}, latency_ms=12.0,
        input_tokens=3, output_tokens=2, cost_usd=0.0, status="complete",
    )
    assert run_result_dict(r) == {
        "agent": "digest",
        "output": {"result": "ok"},
        "status": "complete",
        "latency_ms": 12.0,
        "input_tokens": 3,
        "output_tokens": 2,
        "cost_usd": 0.0,
        "thread_id": None,
        "pending": None,
    }


def test_withheld_dict() -> None:
    assert withheld_dict("digest", input_tokens=4, output_tokens=6) == {
        "agent": "digest",
        "output": {},
        "status": "withheld",
        "latency_ms": 0.0,
        "input_tokens": 4,
        "output_tokens": 6,
        "cost_usd": 0.0,
        "thread_id": None,
        "pending": None,
    }
