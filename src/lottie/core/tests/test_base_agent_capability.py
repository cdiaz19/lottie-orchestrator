"""Rule-11 enforcement through BaseAgent.run: an agent's `_execute` skill calls are
checked against its capability gate; framework skills invoked outside `_execute` are
exempt; the gate is scoped (no ContextVar leak) and nests for sub-agents."""

from __future__ import annotations

from pydantic import BaseModel

from lottie.core.base_agent import BaseAgent
from lottie.core.base_skill import BaseSkill
from lottie.governance.capability import (
    CapabilityDenied,
    CapabilityGate,
    NullCapabilityGate,
    active_capability_gate,
    build_capability_gate,
)
from lottie.llm import MockLLMProvider


class _In(BaseModel):
    q: str


class _Out(BaseModel):
    a: str


class _SkillIn(BaseModel):
    x: int


class _SkillOut(BaseModel):
    y: int


class RetrievalSkill(BaseSkill[_SkillIn, _SkillOut]):
    def _execute(self, data: _SkillIn) -> _SkillOut:
        return _SkillOut(y=data.x + 1)


class SummarizerSkill(BaseSkill[_SkillIn, _SkillOut]):
    def _execute(self, data: _SkillIn) -> _SkillOut:
        return _SkillOut(y=data.x + 2)


class _Agent(BaseAgent[_In, _Out]):
    """Calls whichever skill it is told to, inside its own `_execute`."""

    def __init__(self, skill: BaseSkill[_SkillIn, _SkillOut]) -> None:
        super().__init__(MockLLMProvider(["x"]))
        self._skill = skill

    def _execute(self, data: _In) -> _Out:
        out = self._skill.run(_SkillIn(x=1))
        return _Out(a=f"ok:{out.y}")


def _agent(skill: BaseSkill[_SkillIn, _SkillOut], caps: list[str]) -> _Agent:
    a = _Agent(skill)
    a.set_capability_gate(build_capability_gate(capabilities=caps))
    return a


def test_declared_skill_call_succeeds() -> None:
    a = _agent(RetrievalSkill(enable_benchmarks=False), ["retrieval"])
    assert a.run(_In(q="hi")).a == "ok:2"


def test_undeclared_skill_call_blocked() -> None:
    a = _agent(SummarizerSkill(enable_benchmarks=False), ["retrieval"])
    try:
        a.run(_In(q="hi"))
    except CapabilityDenied as exc:
        assert "summarizer" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected CapabilityDenied")


def test_empty_capabilities_enforces_nothing() -> None:
    a = _agent(SummarizerSkill(enable_benchmarks=False), [])
    assert a.run(_In(q="hi")).a == "ok:3"


def test_gate_does_not_leak_after_run() -> None:
    a = _agent(RetrievalSkill(enable_benchmarks=False), ["retrieval"])
    a.run(_In(q="hi"))
    assert isinstance(active_capability_gate(), NullCapabilityGate)


def test_framework_skill_outside_execute_is_exempt() -> None:
    """A skill invoked in the run WRAPPER (not in `_execute`) must not be gated even
    under a narrow whitelist -- mirrors the security gate calling InputSanitizerSkill."""

    class _GatedAgent(_Agent):
        def __init__(self, skill: BaseSkill[_SkillIn, _SkillOut]) -> None:
            super().__init__(skill)
            self.framework_skill = SummarizerSkill(enable_benchmarks=False)

        def run(self, data: _In) -> _Out:
            # framework skill call BEFORE the _execute window opens -> default Null gate
            self.framework_skill.run(_SkillIn(x=0))
            return super().run(data)

    a = _GatedAgent(RetrievalSkill(enable_benchmarks=False))
    a.set_capability_gate(build_capability_gate(capabilities=["retrieval"]))
    # summarizer is NOT declared, but it runs outside _execute -> exempt; retrieval ok.
    assert a.run(_In(q="hi")).a == "ok:2"


def test_nested_agent_uses_its_own_gate() -> None:
    """A sub-agent enforces its own caps; the parent's gate is restored afterward."""

    class _Parent(BaseAgent[_In, _Out]):
        def __init__(self, child: _Agent) -> None:
            super().__init__(MockLLMProvider(["x"]))
            self._child = child

        def _execute(self, data: _In) -> _Out:
            return self._child.run(data)

    child = _agent(RetrievalSkill(enable_benchmarks=False), ["retrieval"])
    parent = _Parent(child)
    parent.set_capability_gate(CapabilityGate(["something-else"]))
    # child calls retrieval (declared for the child, not the parent) -> allowed
    assert parent.run(_In(q="hi")).a == "ok:2"
    assert isinstance(active_capability_gate(), NullCapabilityGate)
