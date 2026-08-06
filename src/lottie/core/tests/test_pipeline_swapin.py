"""V3 S2a — `run()` executes the middleware chain, in the order `run()` used to hardcode.

The existing ~1390 tests are the real proof that the swap-in is behaviour-preserving.
These tests exist to pin the *ordering itself*, so the invariant is asserted rather than
incidental — and to guard the two deliberate deviations documented in
`runtime.middleware.Order`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from lottie.core.base_agent import BaseAgent, _depth
from lottie.core.middleware import build_chain
from lottie.governance.audit import SqliteAuditLogger
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

    def _set_recall_prefix(self, prefix: str) -> None:
        # V3 S5: recall moved to the memory subsystem; this is the seam it writes through.
        if prefix == "":
            self.log.append("recall_clear")
        super()._set_recall_prefix(prefix)

    def _budgeted_call(self, messages: list[object]) -> str:  # type: ignore[override]
        # E4 S2: reflection is a module now and asks only for a budgeted LLM call.
        # Instrumenting that seam observes the same lifecycle point the old
        # `_maybe_reflect` override did.
        self.log.append("reflect")
        return ""


def _probe(log: list[str]) -> _Probe:
    agent = _Probe(log, llm=MockLLMProvider(responses=["ok"]), enable_benchmarks=False)
    # Reflection is opt-in; enable it so the lifecycle assertion observes that hook.
    agent.set_reflect(enabled=True, namespace="probe")
    return agent


class TestChainComposition:
    def test_every_standard_middleware_is_mounted(self) -> None:
        # 11 after S4: audit left the chain to become an EventBus subscriber.
        chain = build_chain(_probe([]))  # type: ignore[arg-type]
        assert len({m.name for m in chain}) == len(chain) == 11

    def test_audit_is_no_longer_a_middleware(self) -> None:
        # It is an observer, so it subscribes. Its best-effort behaviour is now a
        # property of the bus rather than a try/except it has to remember.
        assert "audit" not in {m.name for m in build_chain(_probe([]))}  # type: ignore[arg-type]

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
    def test_the_chain_is_ordered_as_documented(self) -> None:
        """Asserts the CONTRACT rather than patched internals.

        This test previously instrumented by overriding `_write_audit`,
        `_persist_trajectory` and `_record_session_run` — all of which S4 and S5
        legitimately relocated. The chain's declared order is the thing that actually
        has to hold, and it does not move when ownership does.
        """
        chain = sorted(build_chain(_probe([])), key=lambda m: m.order)  # type: ignore[arg-type]
        assert [m.name for m in chain] == [
            "security_input",
            "policy",
            "cost",
            "session",
            "trajectory",
            "depth",
            "recall",
            "reflect",
            "security_output",
            "verify",
            "capability",
        ]

    def test_the_lifecycle_hooks_still_fire(self) -> None:
        # Recall clears twice by design: once on entry (before loading) and once on the
        # way out, so a disabled or failed recall never leaks a stale prefix.
        log: list[str] = []
        _probe(log).run(_In(task="t"))
        assert log == ["recall_clear", "execute", "verify", "reflect", "recall_clear"]

    def test_verify_runs_before_reflect(self) -> None:
        # `_verify` is fail-closed: a rejected output must not be learned from.
        log: list[str] = []
        _probe(log).run(_In(task="t"))
        assert log.index("verify") < log.index("reflect")

    def test_trajectory_posts_before_session(self) -> None:
        # Post phases unroll in reverse, so a HIGHER order posts earlier.
        by_name = {m.name: m.order for m in build_chain(_probe([]))}  # type: ignore[arg-type]
        assert by_name["trajectory"] > by_name["session"]


class TestAuditRootFlag:
    def test_a_top_level_run_is_audited_root(self, tmp_path: Path) -> None:
        # V3 S4: audit is a SUBSCRIBER, so this observes the real ledger rather than a
        # method override — the mechanism it used to instrument no longer exists.
        agent = _probe([])
        agent._audit = SqliteAuditLogger(tmp_path)
        agent.run(_In(task="t"))
        rows = SqliteAuditLogger(tmp_path).query()
        assert len(rows) == 1 and rows[0].root is True and rows[0].status == "ok"

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
    def test_a_failed_run_still_audits_and_records(self, tmp_path: Path) -> None:
        log: list[str] = []

        class _Failing(_Probe):
            def _execute(self, data: _In) -> _Out:
                log.append("execute")
                raise ValueError("boom")

        agent = _Failing(log, llm=MockLLMProvider(responses=["ok"]), enable_benchmarks=False)
        agent._audit = SqliteAuditLogger(tmp_path)
        with pytest.raises(ValueError):
            agent.run(_In(task="t"))
        rows = SqliteAuditLogger(tmp_path).query()
        assert len(rows) == 1 and rows[0].status == "error"

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
