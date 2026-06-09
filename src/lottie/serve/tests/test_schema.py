from __future__ import annotations

from lottie.serve.schema import AgentInfo, RunResult


def test_agent_info_defaults() -> None:
    info = AgentInfo(name="echo")
    assert info.name == "echo"
    assert info.provider is None


def test_run_result_defaults_and_output() -> None:
    result = RunResult(agent="echo", output={"result": "hi", "n": 3})
    assert result.agent == "echo"
    assert result.output == {"result": "hi", "n": 3}
    assert result.latency_ms == 0.0
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert result.cost_usd == 0.0
