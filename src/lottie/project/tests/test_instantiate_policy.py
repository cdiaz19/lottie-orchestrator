from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.cli import app
from lottie.governance.policy import PolicyDenied
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


def test_instantiate_attaches_policy_that_denies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    cfg_path = demo / "agents" / "echo" / "config.yaml"
    cfg_path.write_text(
        cfg_path.read_text(encoding="utf-8").replace(
            "capabilities: []", "capabilities:\n  - shell"
        ),
        encoding="utf-8",
    )
    (demo / "policies" / "base.yaml").write_text("name: base\ndeny: [shell]\n", encoding="utf-8")
    agent = _build(demo)
    from agents.echo.schema import EchoAgentInput  # type: ignore[import-not-found]

    with pytest.raises(PolicyDenied):
        agent.run(EchoAgentInput(query="hi"))


def test_instantiate_empty_policy_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)  # default base.yaml is empty rules
    agent = _build(demo)
    from agents.echo.schema import EchoAgentInput

    out = agent.run(EchoAgentInput(query="hi"))
    assert out is not None
