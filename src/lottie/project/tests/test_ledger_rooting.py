"""The cost gate and the audit ledger must be the SAME ledger.

The gate READS accrued spend under the project root. An agent constructed without a
`benchmarks_root` WRITES under the cwd. `find_project_root` walks UP, so any command run
from a subdirectory of a project made those two different paths — and a gate polling an
empty ledger admits every run, forever, with no error anywhere.

These tests run with `cwd != root` on purpose, which is the configuration the previous
code documented as "out of scope" while every CLI entry point could reach it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import BaseModel

from lottie.core import BaseAgent
from lottie.governance.audit import SqliteAuditLogger
from lottie.governance.cost import BudgetExceeded
from lottie.llm import LLMResponse, Message
from lottie.llm.base import LLMProvider, TokenUsage
from lottie.project.config import AgentConfig
from lottie.project.discovery import instantiate_agent


class _In(BaseModel):
    query: str


class _Out(BaseModel):
    result: str


class _PricedProvider(LLMProvider):
    def __init__(self, cost: float) -> None:
        self._cost = cost

    @property
    def model(self) -> str:
        return "priced/sim"

    def complete(
        self, messages: list[Message], model_params: Mapping[str, object] | None = None
    ) -> LLMResponse:
        return LLMResponse(
            content="ok",
            usage=TokenUsage(input_tokens=5, output_tokens=5),
            model="priced/sim",
            cost_usd=self._cost,
        )


class _Probe(BaseAgent[_In, _Out]):
    name = "LedgerProbe"

    def _execute(self, data: _In) -> _Out:
        return _Out(result=self.complete([Message(role="user", content=data.query)]).content)


@pytest.fixture
def split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project root, with the cwd set to a SUBDIRECTORY of it.

    This is what `lottie run` sees when invoked from `myproject/agents/`.
    """
    monkeypatch.delenv("LOTTIE_DISABLE_AUDIT", raising=False)
    root = tmp_path / "project"
    (root / "agents" / "probe").mkdir(parents=True)
    (root / "lottie.yaml").write_text("name: demo\n")
    monkeypatch.chdir(root / "agents" / "probe")
    return root


def _agent(root: Path, cost: float, **cfg: object) -> _Probe:
    return instantiate_agent(  # type: ignore[return-value]
        _Probe,  # type: ignore[arg-type]
        llm=_PricedProvider(cost),
        root=root,
        config=AgentConfig.model_validate({"provider": "priced/sim", **cfg}),
        enable_benchmarks=False,
    )


class TestTheLedgerFollowsTheProjectRoot:
    def test_the_run_is_recorded_under_the_root(self, split: Path) -> None:
        # Query by `agent.name`, not a literal: the name is derived from the class, so a
        # hardcoded string would silently sum an empty slice of the ledger.
        agent = _agent(split, 0.04)
        agent.run(_In(query="hi"))
        assert SqliteAuditLogger(split).total_cost(agent.name) == pytest.approx(0.04)

    def test_nothing_is_written_under_the_cwd(self, split: Path) -> None:
        # A second ledger under the cwd is the split itself, not merely untidy.
        _agent(split, 0.04).run(_In(query="hi"))
        assert not (Path.cwd() / ".lottie" / "audit.db").exists()

    def test_the_cwd_is_genuinely_not_the_root(self, split: Path) -> None:
        # Guards the fixture: if these ever coincided the tests above would be vacuous.
        assert Path.cwd() != split


class TestTheBreakerFiresFromASubdirectory:
    def test_cumulative_spend_blocks_the_next_run(self, split: Path) -> None:
        agent = _agent(split, 0.04, budget_usd=0.05)
        agent.run(_In(query="hi"))
        agent.run(_In(query="hi"))  # crosses the budget
        with pytest.raises(BudgetExceeded):
            agent.run(_In(query="hi"))

    def test_committed_spend_shrinks_a_reservation(self, split: Path) -> None:
        # budget 10, max_run_usd 6, $5/run: run 1 settles $5, so 5 + 6 > 10 refuses run 2.
        agent = _agent(split, 5.0, budget_usd=10.0, max_run_usd=6.0)
        agent.run(_In(query="hi"))
        with pytest.raises(BudgetExceeded):
            agent.run(_In(query="hi"))

    def test_an_agent_with_headroom_still_runs(self, split: Path) -> None:
        # The positive control: the refusals above must come from spend, not from a
        # gate that blocks unconditionally once re-rooted.
        agent = _agent(split, 0.01, budget_usd=10.0, max_run_usd=6.0)
        assert agent.run(_In(query="hi")).result == "ok"


class TestSetAuditIsOrderIndependent:
    def test_re_rooting_after_the_bus_is_built_still_takes_effect(self, split: Path) -> None:
        # The bus caches an AuditSubscriber holding the logger, so re-rooting has to
        # invalidate it — otherwise the fix would depend on wiring order.
        agent = _Probe(llm=_PricedProvider(0.04), enable_benchmarks=False)
        agent.event_bus()  # force the bus (and its subscriber) into existence first
        agent.set_audit(SqliteAuditLogger(split))
        agent.run(_In(query="hi"))
        assert SqliteAuditLogger(split).total_cost(agent.name) == pytest.approx(0.04)

    def test_a_disabled_ledger_is_honoured(
        self, split: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOTTIE_DISABLE_AUDIT", "1")
        _agent(split, 0.04).run(_In(query="hi"))
        assert not (split / ".lottie" / "audit.db").exists()
        assert os.getenv("LOTTIE_DISABLE_AUDIT") == "1"
