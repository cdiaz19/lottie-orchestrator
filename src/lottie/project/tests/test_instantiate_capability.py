from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.cli import app
from lottie.governance.capability import CapabilityGate, NullCapabilityGate
from lottie.llm import MockLLMProvider
from lottie.project.config import load_agent_config
from lottie.project.discovery import instantiate_agent, load_agent_class

runner = CliRunner()


def _scaffold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "echo"]).exit_code == 0
    return demo


def _build(demo: Path):  # type: ignore[no-untyped-def]
    cfg = load_agent_config(demo / "agents" / "echo")
    cls = load_agent_class(demo, "echo")
    return instantiate_agent(cls, llm=MockLLMProvider(["hi"]), root=demo, config=cfg)


def test_empty_capabilities_attaches_null_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)  # default config: capabilities: []
    agent = _build(demo)
    assert isinstance(agent._capabilities, NullCapabilityGate)


def test_nonempty_capabilities_attaches_strict_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    cfg_path = demo / "agents" / "echo" / "config.yaml"
    cfg_path.write_text(
        cfg_path.read_text(encoding="utf-8").replace(
            "capabilities: []", "capabilities:\n  - retrieval"
        ),
        encoding="utf-8",
    )
    agent = _build(demo)
    gate = agent._capabilities
    assert isinstance(gate, CapabilityGate)
    assert not isinstance(gate, NullCapabilityGate)
    gate.check("retrieval")  # declared -> ok
    from lottie.governance.capability import CapabilityDenied

    with pytest.raises(CapabilityDenied):
        gate.check("summarizer")
