from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from lottie.cli import app

runner = CliRunner()


def _scaffold_with_evals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "echo"]).exit_code == 0
    (demo / "agents" / "echo" / "evals.yaml").write_text(
        "cases:\n"
        "  - name: greets\n"
        "    input: {query: hello}\n"
        "    expect:\n"
        "      contains: {result: hello}\n",
        encoding="utf-8",
    )
    return demo


def _fake_completion(model: str, messages: object, **kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello world"))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
    )


def test_benchmark_runs_and_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _scaffold_with_evals(tmp_path, monkeypatch)
    monkeypatch.setattr("litellm.completion", _fake_completion)

    result = runner.invoke(app, ["benchmark", "agent", "echo"])
    assert result.exit_code == 0, result.output
    assert "anthropic" in result.output  # provider row

    report_path = demo / ".lottie" / "benchmarks" / "echo-report.json"
    assert report_path.is_file()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["agent"] == "echo"
    assert len(data["providers"]) == 1
    assert data["providers"][0]["accuracy"] == 1.0  # "hello world" contains "hello"


def test_benchmark_compare_two_providers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _scaffold_with_evals(tmp_path, monkeypatch)
    monkeypatch.setattr("litellm.completion", _fake_completion)

    result = runner.invoke(app, ["benchmark", "agent", "echo", "--compare"])
    assert result.exit_code == 0, result.output
    data = json.loads((demo / ".lottie" / "benchmarks" / "echo-report.json").read_text())
    # lottie.yaml default = anthropic/..., fallback = openai/gpt-4o
    assert len(data["providers"]) == 2


def test_benchmark_missing_evals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "echo"]).exit_code == 0
    result = runner.invoke(app, ["benchmark", "agent", "echo"])
    assert result.exit_code != 0
    assert "evals.yaml" in result.output


def test_benchmark_unknown_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    monkeypatch.chdir(tmp_path / "demo")
    result = runner.invoke(app, ["benchmark", "agent", "nope"])
    assert result.exit_code != 0
    assert "nope" in result.output


def test_benchmark_single_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _scaffold_with_evals(tmp_path, monkeypatch)
    monkeypatch.setattr("litellm.completion", _fake_completion)

    result = runner.invoke(app, ["benchmark", "agent", "echo", "--provider", "openai/gpt-4o"])
    assert result.exit_code == 0, result.output
    data = json.loads((demo / ".lottie" / "benchmarks" / "echo-report.json").read_text())
    assert len(data["providers"]) == 1
    assert data["providers"][0]["provider"] == "openai/gpt-4o"


# --- `--learning-delta` (V2 S4) -------------------------------------------------

_LD_SCHEMA = """
from __future__ import annotations
from pydantic import BaseModel


class ProbeInput(BaseModel):
    query: str


class ProbeOutput(BaseModel):
    result: str
"""

_LD_AGENT = """
from __future__ import annotations
from lottie.core import BaseAgent
from lottie.llm import Message

from .schema import ProbeInput, ProbeOutput


class ProbeAgent(BaseAgent[ProbeInput, ProbeOutput]):
    def _execute(self, data: ProbeInput) -> ProbeOutput:
        response = self.complete([Message(role="user", content=data.query)])
        return ProbeOutput(result=response.content)
"""


@pytest.fixture
def ld_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import yaml

    from lottie.llm import MockLLMProvider

    (tmp_path / "lottie.yaml").write_text("name: demo\n")
    agent_dir = tmp_path / "agents" / "probe"
    agent_dir.mkdir(parents=True)
    (tmp_path / "agents" / "__init__.py").write_text("")
    (agent_dir / "__init__.py").write_text("")
    (agent_dir / "agent.py").write_text(_LD_AGENT)
    (agent_dir / "schema.py").write_text(_LD_SCHEMA)
    (agent_dir / "config.yaml").write_text(
        yaml.safe_dump({"provider": "mock/sim", "memory": {"enabled": False}})
    )
    (agent_dir / "evals.yaml").write_text(
        yaml.safe_dump(
            {
                "cases": [
                    {
                        "name": "a",
                        "input": {"query": "q"},
                        "expect": {"contains": {"result": "x"}},
                    }
                ]
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "lottie.benchmark.learning.build_provider",
        lambda model: MockLLMProvider(responses=["x", "x", "x", "x"]),
    )
    return tmp_path


def test_learning_delta_writes_a_report(ld_project: Path) -> None:
    result = CliRunner().invoke(app, ["benchmark", "agent", "probe", "--learning-delta"])
    assert result.exit_code == 0, result.output
    out = ld_project / ".lottie" / "benchmarks" / "probe-learning-delta.json"
    assert out.is_file()


def test_learning_delta_prints_the_verdict(ld_project: Path) -> None:
    result = CliRunner().invoke(app, ["benchmark", "agent", "probe", "--learning-delta"])
    assert "verdict" in result.output


def test_learning_delta_report_is_machine_readable(ld_project: Path) -> None:
    CliRunner().invoke(app, ["benchmark", "agent", "probe", "--learning-delta"])
    payload = json.loads(
        (ld_project / ".lottie" / "benchmarks" / "probe-learning-delta.json").read_text()
    )
    assert payload["agent"] == "probe"
    assert payload["verdict"] in {"improved", "neutral", "regressed"}
    assert len(payload["deltas"]) == 7


def test_learning_delta_rejects_compare(ld_project: Path) -> None:
    # The two flags vary different things; combining them has no coherent meaning.
    result = CliRunner().invoke(
        app, ["benchmark", "agent", "probe", "--learning-delta", "--compare"]
    )
    assert result.exit_code != 0
    assert "one or the other" in result.output


def test_learning_delta_on_an_unknown_agent_fails(ld_project: Path) -> None:
    result = CliRunner().invoke(app, ["benchmark", "agent", "ghost", "--learning-delta"])
    assert result.exit_code != 0
