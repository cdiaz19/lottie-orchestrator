from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from lottie.cli import app
from lottie.core import BaseAgent
from lottie.project.discovery import (
    discover_agents,
    discover_skills,
    load_agent_class,
    load_input_model,
    required_fields,
)

runner = CliRunner()


def _scaffold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "researcher"]).exit_code == 0
    assert runner.invoke(app, ["create", "skill", "cleaner"]).exit_code == 0
    return demo


def test_discover_agents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    agents = discover_agents(demo)
    assert [a.name for a in agents] == ["researcher"]
    assert agents[0].kind == "agent"
    assert agents[0].provider == "anthropic/claude-sonnet-4-6"


def test_discover_skills(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    skills = discover_skills(demo)
    assert [s.name for s in skills] == ["cleaner"]
    assert skills[0].kind == "skill"
    assert skills[0].provider is None


def test_discover_empty_when_no_dir(tmp_path: Path) -> None:
    assert discover_agents(tmp_path) == []
    assert discover_skills(tmp_path) == []


def test_load_agent_class(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    cls = load_agent_class(demo, "researcher")
    assert issubclass(cls, BaseAgent)
    assert cls.__name__ == "ResearcherAgent"


def test_load_agent_class_zero_subclasses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    (demo / "agents" / "researcher" / "agent.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(typer.BadParameter):
        load_agent_class(demo, "researcher")


def test_load_agent_class_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    with pytest.raises(typer.BadParameter):
        load_agent_class(demo, "nope")


def test_load_input_model_and_required_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    model = load_input_model(demo, "researcher")
    assert required_fields(model) == ["query"]


def test_load_input_model_missing_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    (demo / "agents" / "researcher" / "schema.py").write_text("x = 1\n", encoding="utf-8")
    import typer

    with pytest.raises(typer.BadParameter):
        load_input_model(demo, "researcher")
