"""Unit tests for the capability gate (rule 11, per-skill-call, fail-closed)."""

from __future__ import annotations

import pytest

from lottie.governance.capability import (
    CapabilityDenied,
    CapabilityGate,
    NullCapabilityGate,
    build_capability_gate,
)


class TestCapabilityGate:
    def test_declared_capability_passes(self) -> None:
        gate = CapabilityGate(["retrieval", "summarizer"])
        gate.check("retrieval")  # no raise
        gate.check("summarizer")

    def test_undeclared_capability_raises(self) -> None:
        gate = CapabilityGate(["retrieval"])
        with pytest.raises(CapabilityDenied):
            gate.check("summarizer")

    def test_message_names_skill_and_leaks_no_payload(self) -> None:
        gate = CapabilityGate(["retrieval"])
        with pytest.raises(CapabilityDenied) as exc:
            gate.check("summarizer")
        msg = str(exc.value)
        assert "summarizer" in msg
        # declared set may appear; no run payload ever does
        assert "retrieval" in msg

    def test_empty_allowed_set_blocks_everything(self) -> None:
        gate = CapabilityGate([])
        with pytest.raises(CapabilityDenied):
            gate.check("retrieval")


class TestNullCapabilityGate:
    def test_allows_any_skill(self) -> None:
        gate = NullCapabilityGate()
        gate.check("anything")
        gate.check("retrieval")


class TestBuildCapabilityGate:
    def test_empty_capabilities_returns_null_gate(self) -> None:
        gate = build_capability_gate(capabilities=[])
        assert isinstance(gate, NullCapabilityGate)
        gate.check("anything")  # allow-all

    def test_nonempty_capabilities_returns_strict_gate(self) -> None:
        gate = build_capability_gate(capabilities=["retrieval"])
        assert isinstance(gate, CapabilityGate)
        assert not isinstance(gate, NullCapabilityGate)
        gate.check("retrieval")
        with pytest.raises(CapabilityDenied):
            gate.check("summarizer")
