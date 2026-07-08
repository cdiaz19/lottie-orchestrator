from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.cli import app
from lottie.core.security_gate import NullSecurityGate
from lottie.llm import MockLLMProvider
from lottie.project.config import load_agent_config
from lottie.project.discovery import instantiate_agent, load_agent_class
from lottie.serve.security import SecurityGate

runner = CliRunner()


def _scaffold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "echo"]).exit_code == 0
    return demo


def _build(demo: Path, gate: object | None):  # type: ignore[no-untyped-def]
    cfg = load_agent_config(demo / "agents" / "echo")
    cls = load_agent_class(demo, "echo")
    return instantiate_agent(
        cls, llm=MockLLMProvider(["hi"]), root=demo, config=cfg, security_gate=gate  # type: ignore[arg-type]
    )


def test_omitted_security_gate_leaves_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The serve path builds agents WITHOUT a gate -> Null (serve gates externally,
    so no path is double-gated)."""
    demo = _scaffold(tmp_path, monkeypatch)
    agent = _build(demo, None)
    assert isinstance(agent._security, NullSecurityGate)


def test_injected_security_gate_is_attached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI path injects a real SecurityGate -> attached to the BaseAgent chokepoint."""
    demo = _scaffold(tmp_path, monkeypatch)
    gate = SecurityGate()
    agent = _build(demo, gate)
    assert agent._security is gate
