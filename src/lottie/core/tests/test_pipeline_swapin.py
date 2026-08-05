"""V3 S2a — `run()` executes the middleware chain, in the order `run()` used to hardcode.

The existing ~1390 tests are the real proof that the swap-in is behaviour-preserving.
These tests exist to pin the *ordering itself*, so the invariant is asserted rather than
incidental — and to guard the two deliberate deviations documented in
`runtime.middleware.Order`.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from lottie.core.base_agent import BaseAgent, _depth
from lottie.core.middleware import build_chain
from lottie.governance.capability import CapabilityGate, active_capability_gate
from lottie.governance.policy import PolicyDenied, PolicyGate
from lottie.llm import MockLLMProvider


class _In(BaseModel):
    task: str


class _Out(BaseModel):
    answer: str


class _Probe(BaseAgent[_In, _Out]):
    """Records the lifecycle points the chain is supposed to hit, in order."""

    def __init__(self, log: list[str], **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self.log = log

    def _execute(self, data: _In) -> _Out:
        self.log.append("execute")
        return _Out(answer="ok")

    def _verify(self, data: _In, output: _Out) -> None:
        self.log.append("verify")

    def _load_recall(self) -> None:
        self.log.append("recall_load")
        super()._load_recall()

    def _maybe_reflect(self, data: _In, output: _Out) -> None:
        self.log.append("reflect")

    def _write_audit(self, data: _In, output: _Out | None, is_root: bool) -> None:
        self.log.append(f"audit(root={is_root})")

    def _persist_trajectory(self, data: _In, output: _Out | None) -> None:
        self.log.append("trajectory")

    def _record_session_run(self, data: _In) -> None:
        self.log.append("session")


def _probe(log: list[str]) -> _Probe:
    return _Probe(log, llm=MockLLMProvider(responses=["ok"]), enable_benchmarks=False)


class TestChainComposition:
    def test_every_standard_middleware_is_mounted(self) -> None:
        chain = build_chain(_probe([]))  # type: ignore[arg-type]
        assert len({m.name for m in chain}) == len(chain) == 12

    def test_no_two_middleware_share_an_order(self) -> None:
        orders = [m.order for m in build_chain(_probe([]))]  # type: ignore[arg-type]
        assert len(set(orders)) == len(orders)

    def test_the_fail_closed_modules_come_from_their_own_subsystems(self) -> None:
        # V3 S3: security/policy/cost/capability are owned by their subsystems and know
        # nothing about BaseAgent — they are constructed from their gate alone.
        by_name = {m.name: type(m).__module__ for m in build_chain(_probe([]))}  # type: ignore[arg-type]
        assert by_name["security_input"] == "lottie.security.middleware"
        assert by_name["security_output"] == "lottie.security.middleware"
        assert by_name["policy"] == "lottie.governance.middleware"
        assert by_name["cost"] == "lottie.governance.middleware"
        assert by_name["capability"] == "lottie.governance.middleware"


class TestCallOrder:
    def test_the_full_lifecycle_runs_in_the_documented_order(self) -> None:
        log: list[str] = []
        _probe(log).run(_In(task="t"))
        assert log == [
            "recall_load",
            "execute",
            "verify",
            "reflect",
            "audit(root=True)",
            "trajectory",
            "session",
        ]

    def test_verify_runs_before_reflect(self) -> None:
        # `_verify` is fail-closed: a rejected output must not be learned from.
        log: list[str] = []
        _probe(log).run(_In(task="t"))
        assert log.index("verify") < log.index("reflect")

    def test_audit_runs_before_trajectory_and_session(self) -> None:
        log: list[str] = []
        _probe(log).run(_In(task="t"))
        assert log.index("audit(root=True)") < log.index("trajectory") < log.index("session")


class TestAuditRootFlag:
    def test_a_top_level_run_is_root(self) -> None:
        log: list[str] = []
        _probe(log).run(_In(task="t"))
        assert "audit(root=True)" in log

    def test_depth_is_restored_after_the_run(self) -> None:
        _probe([]).run(_In(task="t"))
        assert _depth() == 0

    def test_a_denied_run_is_audited_before_the_depth_increment(self) -> None:
        """The reason DEPTH must sit above COST in the order table.

        `_write_block` reads `_depth() == 0` to decide the root flag. If the depth
        middleware ran before the gates, a denied top-level run would be recorded
        `root=False`.
        """
        seen: list[bool] = []

        class _BlockProbe(_Probe):
            def _write_block(self, data: _In, exc: Exception, status: str) -> None:
                seen.append(_depth() == 0)

        agent = _BlockProbe(
            [], llm=MockLLMProvider(responses=["ok"]), enable_benchmarks=False
        )
        agent.set_policy(PolicyGate(["banned"], allow=set(), deny={"banned"}, escalate=set()))
        with pytest.raises(PolicyDenied):
            agent.run(_In(task="t"))
        assert seen == [True]


class TestCapabilityWindow:
    def test_the_gate_is_active_during_execute(self) -> None:
        seen: list[object] = []

        class _CapProbe(_Probe):
            def _execute(self, data: _In) -> _Out:
                seen.append(active_capability_gate())
                return _Out(answer="ok")

        agent = _CapProbe([], llm=MockLLMProvider(responses=["ok"]), enable_benchmarks=False)
        gate = CapabilityGate(["retrieval"])
        agent.set_capability_gate(gate)
        agent.run(_In(task="t"))
        assert seen == [gate]

    def test_the_gate_is_already_released_during_verify(self) -> None:
        """Why CAPABILITY is the innermost middleware.

        `_verify` is user code that may call a skill, and today it runs with the gate
        already reset. Making capability anything but innermost would silently change
        rule-11 enforcement inside `_verify`.
        """
        seen: list[object] = []

        class _VerifyProbe(_Probe):
            def _verify(self, data: _In, output: _Out) -> None:
                seen.append(active_capability_gate())

        agent = _VerifyProbe([], llm=MockLLMProvider(responses=["ok"]), enable_benchmarks=False)
        gate = CapabilityGate(["retrieval"])
        agent.set_capability_gate(gate)
        agent.run(_In(task="t"))
        assert seen != [gate]  # the run's gate is no longer active


class TestFailurePaths:
    def test_a_failed_run_still_audits_and_records(self) -> None:
        log: list[str] = []

        class _Failing(_Probe):
            def _execute(self, data: _In) -> _Out:
                log.append("execute")
                raise ValueError("boom")

        agent = _Failing(log, llm=MockLLMProvider(responses=["ok"]), enable_benchmarks=False)
        with pytest.raises(ValueError):
            agent.run(_In(task="t"))
        assert "audit(root=True)" in log and "trajectory" in log and "session" in log

    def test_a_failed_run_does_not_reflect(self) -> None:
        # Reflection distils a completed run; a failure has no outcome to learn from.
        log: list[str] = []

        class _Failing(_Probe):
            def _execute(self, data: _In) -> _Out:
                raise ValueError("boom")

        agent = _Failing(log, llm=MockLLMProvider(responses=["ok"]), enable_benchmarks=False)
        with pytest.raises(ValueError):
            agent.run(_In(task="t"))
        assert "reflect" not in log

    def test_recall_is_cleared_even_when_the_run_fails(self) -> None:
        class _Failing(_Probe):
            def _execute(self, data: _In) -> _Out:
                raise ValueError("boom")

        agent = _Failing([], llm=MockLLMProvider(responses=["ok"]), enable_benchmarks=False)
        agent._recall_prefix = "leftover"
        with pytest.raises(ValueError):
            agent.run(_In(task="t"))
        assert agent._recall_prefix == ""
