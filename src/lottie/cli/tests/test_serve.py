from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.cli import app

runner = CliRunner()


def _scaffold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    return tmp_path / "demo"


def test_serve_missing_mcp_shows_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the lazy `from lottie.serve.mcp_server import serve_stdio` to fail,
    # simulating an environment without the [serve] extra installed.
    monkeypatch.setitem(sys.modules, "lottie.serve.mcp_server", None)
    result = runner.invoke(app, ["serve"])
    assert result.exit_code != 0
    assert "lottie-orchestrator[serve]" in result.output


def test_serve_invokes_serve_stdio_with_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    monkeypatch.chdir(demo)
    captured: dict[str, Path] = {}

    # Patch the function the command imports; never enter the real stdio loop.
    monkeypatch.setattr(
        "lottie.serve.mcp_server.serve_stdio",
        lambda root: captured.__setitem__("root", root),
    )
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0, result.output
    assert captured["root"] == demo
