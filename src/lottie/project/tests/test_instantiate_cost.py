from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.cli import app
from lottie.governance.audit import SqliteAuditLogger
from lottie.governance.cost import BudgetExceeded
from lottie.governance.schema import AuditRecord
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


def _seed(demo: Path, agent: str, cost: float) -> None:
    SqliteAuditLogger(demo).log(
        AuditRecord(
            ts="2026-06-14T10:00:00+00:00", agent=agent, provider="mock/x", status="ok",
            root=True, input_sha256="a" * 64, output_sha256="b" * 64, input_tokens=0,
            output_tokens=0, cost_usd=cost, latency_ms=1.0, error=None,
        )
    )


def test_instantiate_attaches_budget_that_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LOTTIE_DISABLE_AUDIT", raising=False)  # audit ledger must be readable
    demo = _scaffold(tmp_path, monkeypatch)
    cfg_path = demo / "agents" / "echo" / "config.yaml"
    cfg_path.write_text(
        cfg_path.read_text(encoding="utf-8") + "budget_usd: 0.05\n", encoding="utf-8"
    )
    cfg = load_agent_config(demo / "agents" / "echo")
    cls = load_agent_class(demo, "echo")
    agent = instantiate_agent(cls, llm=MockLLMProvider(["hi"]), root=demo, config=cfg)
    _seed(demo, agent.name, 0.05)  # prior spend at budget, under this agent's metrics name
    from agents.echo.schema import EchoAgentInput  # type: ignore[import-not-found]

    with pytest.raises(BudgetExceeded):
        agent.run(EchoAgentInput(query="hi"))


def test_instantiate_no_budget_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    cfg = load_agent_config(demo / "agents" / "echo")
    cls = load_agent_class(demo, "echo")
    agent = instantiate_agent(cls, llm=MockLLMProvider(["hi"]), root=demo, config=cfg)
    from agents.echo.schema import EchoAgentInput

    assert agent.run(EchoAgentInput(query="hi")) is not None
