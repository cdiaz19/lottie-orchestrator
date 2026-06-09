from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.llm import LLMResponse, Message, MockLLMProvider
from lottie.llm.base import LLMProvider
from lottie.serve.security import SecurityGate
from lottie.serve.service import (
    AgentExecutionError,
    AgentLoadError,
    AgentNotFoundError,
    AgentService,
    InvalidInputError,
)

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
    # `lottie init` ships a runnable hello agent, so it lists alongside `echo`.
    assert [i.name for i in infos] == ["echo", "hello"]
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


class _SpyGate(SecurityGate):
    """Records the order of gate calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def check_input(self, text: str) -> str:
        self.calls.append("in")
        return super().check_input(text)

    def check_output(self, text: str) -> str:
        self.calls.append("out")
        return super().check_output(text)


def _mock_provider(monkeypatch: pytest.MonkeyPatch, response: str = "hello world") -> None:
    """Patch build_provider in the service module to return a MockLLMProvider."""
    monkeypatch.setattr(
        "lottie.serve.service.build_provider",
        lambda name: MockLLMProvider([response]),
    )


def test_run_agent_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    _mock_provider(monkeypatch)
    svc = AgentService(demo)
    result = svc.run_agent("echo", {"query": "hi"})
    assert result.agent == "echo"
    assert result.output == {"result": "hello world"}
    assert result.latency_ms >= 0.0  # metrics populated from last_metrics


def test_run_agent_unknown_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    svc = AgentService(demo)
    with pytest.raises(AgentNotFoundError):
        svc.run_agent("nope", {"query": "hi"})


def test_run_agent_invalid_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    _mock_provider(monkeypatch)
    svc = AgentService(demo)
    with pytest.raises(InvalidInputError):
        svc.run_agent("echo", {"wrong": "field"})  # echo Input requires `query`


def test_run_agent_execution_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "lottie.serve.service.build_provider", lambda name: _BoomProvider()
    )
    svc = AgentService(demo)
    with pytest.raises(AgentExecutionError) as exc_info:
        svc.run_agent("echo", {"query": "hi"})
    assert exc_info.value.__cause__ is not None  # original error chained


def test_run_agent_gate_called_input_then_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    _mock_provider(monkeypatch)
    gate = _SpyGate()
    svc = AgentService(demo, gate=gate)
    svc.run_agent("echo", {"query": "hi"})
    assert gate.calls == ["in", "out"]


def test_run_agent_provider_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    captured: list[str] = []

    def fake_build(name: str) -> MockLLMProvider:
        captured.append(name)
        return MockLLMProvider(["ok"])

    monkeypatch.setattr("lottie.serve.service.build_provider", fake_build)
    svc = AgentService(demo)

    svc.run_agent("echo", {"query": "hi"}, provider="openai/gpt-4o")
    assert captured == ["openai/gpt-4o"]

    captured.clear()
    svc.run_agent("echo", {"query": "hi"})  # falls back to config provider
    assert captured[0] is not None
    assert "anthropic" in captured[0]


def test_public_exports() -> None:
    import lottie.serve as serve

    for symbol in (
        "AgentInfo",
        "RunResult",
        "SecurityGate",
        "AgentService",
        "ServeError",
        "AgentNotFoundError",
        "InvalidInputError",
        "AgentExecutionError",
        "AgentLoadError",
    ):
        assert hasattr(serve, symbol), symbol


def test_run_agent_bad_config_raises_load_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    # provider must be a string; a list fails AgentConfig validation, which
    # load_agent_config surfaces as typer.BadParameter — the core must translate it.
    (demo / "agents" / "echo" / "config.yaml").write_text(
        "provider: [not, a, string]\n", encoding="utf-8"
    )
    svc = AgentService(demo)
    with pytest.raises(AgentLoadError):
        svc.run_agent("echo", {"query": "hi"})


def test_run_agent_broken_schema_raises_load_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    # A SyntaxError in schema.py escapes the importer raw; the core must translate it.
    (demo / "agents" / "echo" / "schema.py").write_text(
        "def (this is not valid python\n", encoding="utf-8"
    )
    svc = AgentService(demo)
    with pytest.raises(AgentLoadError):
        svc.run_agent("echo", {"query": "hi"})
