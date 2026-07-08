"""Security gate (rules 8 & 9) on the BaseAgent chokepoint: input screened before
`_execute`, output screened after; default construction is ungated; the gate's checks
run OUTSIDE the capability window (framework-skill exemption preserved)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from lottie.core.base_agent import BaseAgent
from lottie.core.security_gate import NullSecurityGate
from lottie.governance.audit import SqliteAuditLogger
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


class _BlockInput:
    """Gate that rejects any input containing 'BADIN'."""

    def check_input(self, text: str) -> None:
        if "BADIN" in text:
            raise ValueError("input rejected")

    def check_output(self, text: str) -> None:
        return


class _BlockOutput:
    def check_input(self, text: str) -> None:
        return

    def check_output(self, text: str) -> None:
        if "SECRET" in text:
            raise ValueError("output withheld")


def _agent(tmp_path: Path) -> _Spy:
    return _Spy(MockLLMProvider(["x"]), SqliteAuditLogger(tmp_path))


class TestNullSecurityGate:
    def test_noops(self) -> None:
        g = NullSecurityGate()
        g.check_input("anything")
        g.check_output("anything")


class TestBaseAgentSecurity:
    def test_default_is_ungated(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        assert isinstance(agent._security, NullSecurityGate)
        assert agent.run(_In(q="hi")).a == "ok:hi"

    def test_bad_input_blocked_before_execute(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        agent.set_security_gate(_BlockInput())
        with pytest.raises(ValueError, match="input rejected"):
            agent.run(_In(q="BADIN"))
        assert agent.ran is False  # _execute never reached

    def test_bad_output_withheld_after_execute_and_audited(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        agent.set_security_gate(_BlockOutput())

        class _SecretSpy(_Spy):
            def _execute(self, data: _In) -> _Out:
                self.ran = True
                return _Out(a="SECRET value")

        secret_agent = _SecretSpy(MockLLMProvider(["x"]), SqliteAuditLogger(tmp_path))
        secret_agent.set_security_gate(_BlockOutput())
        with pytest.raises(ValueError, match="output withheld"):
            secret_agent.run(_In(q="hi"))
        assert secret_agent.ran is True  # the run executed; output withheld at the boundary
        rows = SqliteAuditLogger(tmp_path).query()
        assert rows and rows[0].status == "ok"  # executed run is still audited

    def test_clean_run_passes_both_gates(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        agent.set_security_gate(_BlockInput())  # 'ok:hi' contains no BADIN/SECRET
        assert agent.run(_In(q="hi")).a == "ok:hi"
