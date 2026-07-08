"""BaseSkill capability-name resolution + rule-11 enforcement in BaseSkill.run."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from lottie.core import BaseSkill
from lottie.governance.capability import (
    CapabilityDenied,
    CapabilityGate,
    NullCapabilityGate,
    _active_capabilities,
)


class _In(BaseModel):
    x: int


class _Out(BaseModel):
    y: int


class RetrievalSkill(BaseSkill[_In, _Out]):
    def _execute(self, data: _In) -> _Out:
        return _Out(y=data.x)


class SummarizerSkill(BaseSkill[_In, _Out]):
    def _execute(self, data: _In) -> _Out:
        return _Out(y=data.x)


class Plain(BaseSkill[_In, _Out]):  # no "Skill" suffix
    def _execute(self, data: _In) -> _Out:
        return _Out(y=data.x)


class Explicit(BaseSkill[_In, _Out]):
    capability_name = "custom-cap"

    def _execute(self, data: _In) -> _Out:
        return _Out(y=data.x)


class TestCapabilityNameResolution:
    def test_strips_skill_suffix_and_lowercases(self) -> None:
        assert RetrievalSkill.resolved_capability_name() == "retrieval"
        assert SummarizerSkill.resolved_capability_name() == "summarizer"

    def test_no_suffix_lowercases_full_name(self) -> None:
        assert Plain.resolved_capability_name() == "plain"

    def test_explicit_override_wins(self) -> None:
        assert Explicit.resolved_capability_name() == "custom-cap"


class TestBaseSkillEnforcement:
    def test_default_null_gate_allows(self) -> None:
        # No active agent context -> default NullCapabilityGate -> allowed.
        assert RetrievalSkill(enable_benchmarks=False).run(_In(x=3)).y == 3

    def test_active_strict_gate_allows_declared(self) -> None:
        token = _active_capabilities.set(CapabilityGate(["retrieval"]))
        try:
            assert RetrievalSkill(enable_benchmarks=False).run(_In(x=5)).y == 5
        finally:
            _active_capabilities.reset(token)

    def test_active_strict_gate_blocks_undeclared(self) -> None:
        token = _active_capabilities.set(CapabilityGate(["retrieval"]))
        try:
            with pytest.raises(CapabilityDenied):
                SummarizerSkill(enable_benchmarks=False).run(_In(x=5))
        finally:
            _active_capabilities.reset(token)

    def test_null_gate_explicitly_set_allows(self) -> None:
        token = _active_capabilities.set(NullCapabilityGate())
        try:
            SummarizerSkill(enable_benchmarks=False).run(_In(x=1))
        finally:
            _active_capabilities.reset(token)
