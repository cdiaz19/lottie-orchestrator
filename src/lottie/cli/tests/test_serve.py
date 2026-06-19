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


def test_serve_port_runs_uvicorn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`lottie serve --port N` builds the OpenAI app and hands it to uvicorn.run."""
    pytest.importorskip("starlette")
    from lottie.cli import app

    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "demo"])
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)

    captured: dict[str, object] = {}

    def fake_run(application: object, **kwargs: object) -> None:
        captured["app"] = application
        captured["port"] = kwargs.get("port")

    monkeypatch.setattr("uvicorn.run", fake_run)
    result = runner.invoke(app, ["serve", "--port", "8123"])
    assert result.exit_code == 0
    assert captured["port"] == 8123
    served = captured["app"]
    assert served is not None
    paths = {r.path for r in served.routes}  # type: ignore[attr-defined]
    assert "/v1/agents" in paths        # REST group present -> build_http_app
    assert "/v1/chat/completions" in paths  # OpenAI group present


def test_serve_no_port_uses_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    """No --port keeps the existing stdio MCP path."""
    from lottie.cli import app

    called: dict[str, bool] = {"stdio": False}
    monkeypatch.setattr(
        "lottie.serve.mcp_server.serve_stdio",
        lambda root: called.__setitem__("stdio", True),
    )
    monkeypatch.setattr(
        "lottie.project.config.find_project_root", lambda: Path(".")
    )
    runner.invoke(app, ["serve"])
    assert called["stdio"] is True
