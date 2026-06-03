from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.serve.service import AgentService

runner = CliRunner()


def _scaffold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a real project with one generated `echo` agent on disk."""
    from lottie.cli import app

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "echo"]).exit_code == 0
    return demo


def test_list_agents_returns_name_and_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    svc = AgentService(demo)
    infos = svc.list_agents()
    assert [i.name for i in infos] == ["echo"]
    assert infos[0].provider is not None
    assert "anthropic" in infos[0].provider


def test_list_agents_does_not_import_user_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    broken = demo / "agents" / "broken"
    broken.mkdir()
    (broken / "agent.py").write_text("this is !!! not valid python", encoding="utf-8")
    svc = AgentService(demo)
    names = {i.name for i in svc.list_agents()}
    assert {"echo", "broken"} <= names  # broken lists despite unimportable agent.py
