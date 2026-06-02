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
