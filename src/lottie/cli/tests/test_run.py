from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from lottie.cli import app

runner = CliRunner()


def _scaffold_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "echo"]).exit_code == 0
    return demo


def _fake_completion_factory(
    captured: dict[str, object],
) -> Callable[..., SimpleNamespace]:
    def fake_completion(model: str, messages: object, **kwargs: object) -> SimpleNamespace:
        captured["model"] = model
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hello world"))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
        )

    return fake_completion


def test_run_executes_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _scaffold_agent(tmp_path, monkeypatch)
    captured: dict[str, object] = {}
    monkeypatch.setattr("litellm.completion", _fake_completion_factory(captured))

    result = runner.invoke(app, ["run", "echo", "--input", '{"query": "hi"}'])
    assert result.exit_code == 0, result.output
    assert "hello world" in result.output
    assert captured["model"] == "anthropic/claude-sonnet-4-6"


def test_run_provider_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _scaffold_agent(tmp_path, monkeypatch)
    captured: dict[str, object] = {}
    monkeypatch.setattr("litellm.completion", _fake_completion_factory(captured))

    result = runner.invoke(
        app, ["run", "echo", "--input", '{"query": "hi"}', "--provider", "openai/gpt-4o"]
    )
    assert result.exit_code == 0, result.output
    assert captured["model"] == "openai/gpt-4o"


def test_run_unknown_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _scaffold_agent(tmp_path, monkeypatch)
    result = runner.invoke(app, ["run", "nope", "--input", "{}"])
    assert result.exit_code != 0
    assert "nope" in result.output


def test_run_malformed_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _scaffold_agent(tmp_path, monkeypatch)
    result = runner.invoke(app, ["run", "echo", "--input", "{not json"])
    assert result.exit_code != 0
    assert "echo" in result.output


def test_run_missing_required_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _scaffold_agent(tmp_path, monkeypatch)
    result = runner.invoke(app, ["run", "echo"])
    assert result.exit_code != 0
    assert "query" in result.output


def test_run_surfaces_agent_error_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _scaffold_agent(tmp_path, monkeypatch)

    def _boom(model: str, messages: object, **kwargs: object) -> object:
        raise RuntimeError("no API key")

    monkeypatch.setattr("litellm.completion", _boom)
    result = runner.invoke(app, ["run", "echo", "--input", '{"query": "hi"}'])
    assert result.exit_code != 0
    assert "failed" in result.output
