"""Post-run episodic write-back.

Best-effort and OFF by default, mirroring `_write_audit`: this must never fail, slow,
or alter a run. Unlike reflection it makes no LLM call, so it has no budget interaction.
"""

from __future__ import annotations

import json
import warnings

import pytest
from pydantic import BaseModel

from lottie.core.base_agent import BaseAgent
from lottie.governance.policy import PolicyDenied, PolicyGate
from lottie.llm import MockLLMProvider
from lottie.memory.base import MemoryClient
from lottie.memory.mock import MockMemoryClient
from lottie.memory.schema import MemoryQuery, MemoryRecord, MemoryTier


class _Input(BaseModel):
    task: str


class _Output(BaseModel):
    answer: str


class _Agent(BaseAgent[_Input, _Output]):
    def _execute(self, data: _Input) -> _Output:
        return _Output(answer=data.task.upper())


class _Failing(BaseAgent[_Input, _Output]):
    def _execute(self, data: _Input) -> _Output:
        raise ValueError("boom")


def _deny_all() -> PolicyGate:
    return PolicyGate(["banned"], allow=set(), deny={"banned"}, escalate=set())


class _BrokenMemory(MockMemoryClient):
    def remember(self, record: MemoryRecord) -> str:
        raise RuntimeError("disk on fire")


def _agent(
    cls: type[BaseAgent[_Input, _Output]] = _Agent,
    *,
    memory: MemoryClient | None = None,
    enabled: bool = True,
    max_chars: int = 4000,
) -> BaseAgent[_Input, _Output]:
    agent = cls(
        llm=MockLLMProvider(responses=["ok"]),
        memory=memory or MockMemoryClient(),
        enable_benchmarks=False,
    )
    agent.set_trajectory(enabled=enabled, namespace="ns", max_chars=max_chars)
    return agent


def _episodic(agent: BaseAgent[_Input, _Output]) -> list[str]:
    query = MemoryQuery(text="", namespace="ns", tier=MemoryTier.EPISODIC, limit=100)
    return [hit.record.content for hit in agent.memory.recall(query).hits]


class TestOptIn:
    def test_disabled_by_default_writes_nothing(self) -> None:
        agent = _Agent(
            llm=MockLLMProvider(responses=["ok"]),
            memory=MockMemoryClient(),
            enable_benchmarks=False,
        )
        agent.run(_Input(task="hi"))
        assert _episodic(agent) == []

    def test_explicitly_disabled_writes_nothing(self) -> None:
        agent = _agent(enabled=False)
        agent.run(_Input(task="hi"))
        assert _episodic(agent) == []


class TestSuccessfulRun:
    def test_persists_one_record(self) -> None:
        agent = _agent()
        agent.run(_Input(task="hi"))
        assert len(_episodic(agent)) == 1

    def test_record_is_a_parseable_trajectory(self) -> None:
        agent = _agent()
        agent.run(_Input(task="hi"))
        payload = json.loads(_episodic(agent)[0])
        assert payload["success"] is True
        assert "hi" in payload["task"]
        assert "HI" in payload["outcome"]

    def test_record_carries_run_metrics(self) -> None:
        agent = _agent()
        agent.run(_Input(task="hi"))
        payload = json.loads(_episodic(agent)[0])
        assert payload["latency_ms"] >= 0.0
        assert "input_tokens" in payload

    def test_two_runs_persist_two_records(self) -> None:
        agent = _agent()
        agent.run(_Input(task="hi"))
        agent.run(_Input(task="hi"))
        assert len(_episodic(agent)) == 2

    def test_the_run_output_is_unaffected(self) -> None:
        agent = _agent()
        assert agent.run(_Input(task="hi")).answer == "HI"


class TestFailedRun:
    def test_a_failed_run_is_still_persisted(self) -> None:
        # Failures are the more useful half of the corpus for reflection.
        agent = _agent(_Failing)
        with pytest.raises(ValueError):
            agent.run(_Input(task="hi"))
        assert len(_episodic(agent)) == 1

    def test_failure_is_recorded_as_unsuccessful_with_the_error(self) -> None:
        agent = _agent(_Failing)
        with pytest.raises(ValueError):
            agent.run(_Input(task="hi"))
        payload = json.loads(_episodic(agent)[0])
        assert payload["success"] is False
        assert "boom" in (payload["error"] or "")

    def test_failure_persists_no_outcome(self) -> None:
        agent = _agent(_Failing)
        with pytest.raises(ValueError):
            agent.run(_Input(task="hi"))
        assert json.loads(_episodic(agent)[0])["outcome"] == ""


class TestBlockedRun:
    def test_a_policy_denied_run_persists_nothing(self) -> None:
        # The gates raise before `run` enters its try/finally, so there is no trajectory
        # to record — the work never started.
        agent = _agent()
        agent.set_policy(_deny_all())
        with pytest.raises(PolicyDenied):
            agent.run(_Input(task="hi"))
        assert _episodic(agent) == []


class TestClipping:
    def test_oversized_task_is_clipped(self) -> None:
        agent = _agent(max_chars=10)
        agent.run(_Input(task="x" * 500))
        payload = json.loads(_episodic(agent)[0])
        assert payload["task"].endswith("…[clipped]")

    def test_clipped_record_stays_bounded(self) -> None:
        agent = _agent(max_chars=10)
        agent.run(_Input(task="x" * 5000))
        assert len(_episodic(agent)[0]) < 500


class TestBestEffort:
    def test_a_store_failure_never_fails_the_run(self) -> None:
        agent = _agent(memory=_BrokenMemory())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert agent.run(_Input(task="hi")).answer == "HI"

    def test_a_store_failure_warns(self) -> None:
        agent = _agent(memory=_BrokenMemory())
        with pytest.warns(UserWarning, match="trajectory"):
            agent.run(_Input(task="hi"))

    def test_a_store_failure_does_not_suppress_a_run_error(self) -> None:
        agent = _agent(_Failing, memory=_BrokenMemory())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="boom"):
                agent.run(_Input(task="hi"))


class TestNoBudgetInteraction:
    def test_persistence_makes_no_llm_call(self) -> None:
        # Reflection routes through self.complete and counts against the token cap.
        # Trajectory persistence must not: it is pure serialization.
        agent = _agent()
        agent.run(_Input(task="hi"))
        assert agent.last_metrics is not None
        assert agent.last_metrics.input_tokens == 0
