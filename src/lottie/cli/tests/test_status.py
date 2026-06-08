from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.cli import app

runner = CliRunner()


def test_status_lists_units(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    monkeypatch.chdir(tmp_path / "demo")
    assert runner.invoke(app, ["create", "agent", "researcher"]).exit_code == 0
    assert runner.invoke(app, ["create", "skill", "cleaner"]).exit_code == 0

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.output
    assert "demo" in result.output
    assert "researcher" in result.output
    assert "cleaner" in result.output


def test_status_empty_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    monkeypatch.chdir(tmp_path / "demo")
    # init ships a hello agent; remove it to exercise the empty-project rendering.
    shutil.rmtree(tmp_path / "demo" / "agents" / "hello")
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.output
    assert "No agents" in result.output


def test_status_outside_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["status"])
    assert result.exit_code != 0
