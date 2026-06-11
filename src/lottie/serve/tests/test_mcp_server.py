from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.llm import LLMResponse, Message, MockLLMProvider
from lottie.llm.base import LLMProvider

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


class _BoomProvider(LLMProvider):
    """Provider whose complete() always raises — to force an execution error."""

    @property
    def model(self) -> str:
        return "boom/boom"

    def complete(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> LLMResponse:
        raise RuntimeError("boom")


async def test_call_tool_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mcp.shared.memory import create_connected_server_and_client_session

    from lottie.serve.mcp_server import build_mcp_server

    demo = _scaffold(tmp_path, monkeypatch)
    _mock_provider(monkeypatch)
    server = build_mcp_server(demo)
    async with create_connected_server_and_client_session(server) as client:
        res = await client.call_tool("echo", {"query": "hi"})

    assert res.isError is False
    assert res.structuredContent == {"result": "hello world"}
    texts = [c.text for c in res.content if c.type == "text"]
    assert any("[lottie]" in t for t in texts)  # metrics line present


async def test_call_tool_invalid_input_is_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mcp.shared.memory import create_connected_server_and_client_session

    from lottie.serve.mcp_server import build_mcp_server

    demo = _scaffold(tmp_path, monkeypatch)
    _mock_provider(monkeypatch)
    server = build_mcp_server(demo)
    async with create_connected_server_and_client_session(server) as client:
        res = await client.call_tool("echo", {"wrong": "field"})

    assert res.isError is True  # InvalidInputError → isError


async def test_call_tool_execution_error_is_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mcp.shared.memory import create_connected_server_and_client_session

    from lottie.serve.mcp_server import build_mcp_server

    demo = _scaffold(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "lottie.serve.service.build_provider", lambda name: _BoomProvider()
    )
    server = build_mcp_server(demo)
    async with create_connected_server_and_client_session(server) as client:
        res = await client.call_tool("echo", {"query": "hi"})

    assert res.isError is True  # AgentExecutionError → isError
