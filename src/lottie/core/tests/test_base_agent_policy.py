from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from lottie.core.base_agent import BaseAgent
from lottie.governance.audit import SqliteAuditLogger
from lottie.governance.policy import (
    NullPolicyGate,
    PolicyDenied,
    PolicyEscalation,
    PolicyGate,
)
from lottie.llm import MockLLMProvider


class _In(BaseModel):
    q: str


class _Out(BaseModel):
    a: str


class _Spy(BaseAgent[_In, _Out]):
    def __init__(self, llm: object, audit: object) -> None:
        super().__init__(llm, audit=audit)  # type: ignore[arg-type]
        self.ran = False

    def _execute(self, data: _In) -> _Out:
        self.ran = True
        return _Out(a=f"ok:{data.q}")


def _agent(tmp_path: Path) -> _Spy:
    return _Spy(MockLLMProvider(["x"]), SqliteAuditLogger(tmp_path))


def test_denied_run_blocks_before_execute_and_audits(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent.set_policy(PolicyGate(["shell"], allow=set(), deny={"shell"}, escalate=set()))
    with pytest.raises(PolicyDenied):
        agent.run(_In(q="hi"))
    assert agent.ran is False  # _execute never reached
    rows = SqliteAuditLogger(tmp_path).query()
    assert len(rows) == 1 and rows[0].status == "denied" and rows[0].output_sha256 is None


def test_escalated_run_audits_escalated(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent.set_policy(PolicyGate(["fs"], allow=set(), deny=set(), escalate={"fs"}))
    with pytest.raises(PolicyEscalation):
        agent.run(_In(q="hi"))
    rows = SqliteAuditLogger(tmp_path).query()
    assert rows[0].status == "escalated"


def test_default_null_gate_runs_normally(tmp_path: Path) -> None:
    agent = _agent(tmp_path)  # no set_policy → NullPolicyGate default
    out = agent.run(_In(q="hi"))
    assert out.a == "ok:hi" and agent.ran is True
    rows = SqliteAuditLogger(tmp_path).query()
    assert rows[0].status == "ok"


def test_default_policy_is_null_gate(tmp_path: Path) -> None:
    assert isinstance(_agent(tmp_path)._policy, NullPolicyGate)
