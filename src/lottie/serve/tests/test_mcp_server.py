from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.llm import MockLLMProvider

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


def _mock_provider(monkeypatch: pytest.MonkeyPatch, response: str = "hello world") -> None:
    """Patch build_provider in the service module to return a MockLLMProvider."""
    monkeypatch.setattr(
        "lottie.serve.service.build_provider",
        lambda name: MockLLMProvider([response]),
    )


async def test_list_tools_one_per_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mcp.shared.memory import create_connected_server_and_client_session

    from lottie.project.discovery import load_input_model
    from lottie.serve.mcp_server import build_mcp_server

    demo = _scaffold(tmp_path, monkeypatch)
    server = build_mcp_server(demo)
    async with create_connected_server_and_client_session(server) as client:
        result = await client.list_tools()

    names = {t.name for t in result.tools}
    assert {"echo", "hello"} <= names  # `init` ships a hello agent; we added echo
    echo = next(t for t in result.tools if t.name == "echo")
    assert echo.inputSchema == load_input_model(demo, "echo").model_json_schema()
    assert echo.description  # non-empty (first system-prompt line)


async def test_broken_agent_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mcp.shared.memory import create_connected_server_and_client_session

    from lottie.serve.mcp_server import build_mcp_server

    demo = _scaffold(tmp_path, monkeypatch)
    broken = demo / "agents" / "broken"
    broken.mkdir()
    (broken / "agent.py").write_text("!!! not valid python", encoding="utf-8")

    server = build_mcp_server(demo)
    async with create_connected_server_and_client_session(server) as client:
        result = await client.list_tools()

    names = {t.name for t in result.tools}
    assert "broken" not in names  # unimportable agent skipped, not fatal
    assert "echo" in names        # healthy agents still register
