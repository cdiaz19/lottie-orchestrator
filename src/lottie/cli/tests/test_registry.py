from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.cli import app

runner = CliRunner()


def _scaffold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "researcher"]).exit_code == 0
    assert runner.invoke(app, ["create", "skill", "cleaner"]).exit_code == 0
    return demo


def test_list_agents_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    monkeypatch.chdir(tmp_path / "demo")
    result = runner.invoke(app, ["list", "agents"])
    assert result.exit_code == 0, result.output
    assert "No agents" in result.output


def test_list_agents_populated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _scaffold(tmp_path, monkeypatch)
    result = runner.invoke(app, ["list", "agents"])
    assert result.exit_code == 0, result.output
    assert "researcher" in result.output
    assert "anthropic" in result.output


def test_list_skills_populated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _scaffold(tmp_path, monkeypatch)
    result = runner.invoke(app, ["list", "skills"])
    assert result.exit_code == 0, result.output
    assert "cleaner" in result.output
    assert "CleanerSkillInput" in result.output
    assert "CleanerSkillOutput" in result.output


def test_list_skills_broken_schema_degrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    (demo / "skills" / "cleaner" / "schema.py").write_text(
        "raise RuntimeError('boom')\n", encoding="utf-8"
    )
    result = runner.invoke(app, ["list", "skills"])
    assert result.exit_code == 0, result.output
    assert "cleaner" in result.output
    assert "—" in result.output


def test_list_outside_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["list", "agents"]).exit_code != 0
