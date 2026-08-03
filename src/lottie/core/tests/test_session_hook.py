"""Session hooks on BaseAgent: resume, incremental progress, and run history."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from pydantic import BaseModel

from lottie.core.base_agent import BaseAgent
from lottie.llm import MockLLMProvider
from lottie.session.store import SessionStore


class _Input(BaseModel):
    task: str


class _Output(BaseModel):
    answer: str


class _Stepper(BaseAgent[_Input, _Output]):
    """Advances a counter and records it, the way a resumable agent would."""

    def _execute(self, data: _Input) -> _Output:
        # `progress` is dict[str, object] by design — it holds whatever the agent needs.
        # Narrowing on read is the consumer's job, and is also the right instinct: this
        # data crossed a process boundary and is not to be trusted blindly.
        raw = self.session_progress.get("step", 0)
        step = raw if isinstance(raw, int) else 0
        self.save_progress(step=step + 1)
        return _Output(answer=f"step {step + 1}")


class _Failing(BaseAgent[_Input, _Output]):
    def _execute(self, data: _Input) -> _Output:
        self.save_progress(reached="halfway")
        raise ValueError("boom")


def _agent(cls: type[BaseAgent[_Input, _Output]] = _Stepper) -> BaseAgent[_Input, _Output]:
    return cls(llm=MockLLMProvider(responses=["ok"]), enable_benchmarks=False)


class TestOptIn:
    def test_no_session_means_empty_progress(self) -> None:
        assert _agent().session_progress == {}

    def test_save_progress_without_a_session_is_a_no_op(self) -> None:
        agent = _agent()
        agent.save_progress(step=5)  # must not raise
        assert agent.session_progress == {}

    def test_a_run_without_a_session_writes_nothing(self, tmp_path: Path) -> None:
        agent = _agent()
        agent.run(_Input(task="t"))
        assert SessionStore(tmp_path).list() == []


class TestProgress:
    def test_progress_persists_across_agents(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        first = _agent()
        first.set_session(store, "s1")
        first.run(_Input(task="t"))

        # A NEW agent instance — stands in for a separate process.
        second = _agent()
        second.set_session(store, "s1")
        assert second.session_progress == {"step": 1}

    def test_successive_runs_advance(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        for _ in range(3):
            agent = _agent()
            agent.set_session(store, "s1")
            agent.run(_Input(task="t"))
        assert store.require("s1").progress == {"step": 3}

    def test_the_output_reflects_resumed_state(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        a = _agent()
        a.set_session(store, "s1")
        a.run(_Input(task="t"))
        b = _agent()
        b.set_session(store, "s1")
        assert b.run(_Input(task="t")).answer == "step 2"

    def test_progress_is_merged_not_replaced(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        agent = _agent()
        agent.set_session(store, "s1")
        agent.save_progress(a=1)
        agent.save_progress(b=2)
        assert store.require("s1").progress == {"a": 1, "b": 2}

    def test_progress_survives_a_failed_run(self, tmp_path: Path) -> None:
        # The point of persisting per call: a run that dies halfway must leave behind
        # what it had already achieved.
        store = SessionStore(tmp_path)
        agent = _agent(_Failing)
        agent.set_session(store, "s1")
        with pytest.raises(ValueError):
            agent.run(_Input(task="t"))
        assert store.require("s1").progress == {"reached": "halfway"}

    def test_session_progress_is_a_copy(self, tmp_path: Path) -> None:
        # Mutating the returned dict must not silently edit session state.
        store = SessionStore(tmp_path)
        agent = _agent()
        agent.set_session(store, "s1")
        agent.save_progress(step=1)
        agent.session_progress["step"] = 99
        assert agent.session_progress == {"step": 1}


class TestRunHistory:
    def test_a_run_is_recorded(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        agent = _agent()
        agent.set_session(store, "s1")
        agent.run(_Input(task="t"))
        assert len(store.require("s1").runs) == 1

    def test_history_accumulates(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        for _ in range(3):
            agent = _agent()
            agent.set_session(store, "s1")
            agent.run(_Input(task="t"))
        assert len(store.require("s1").runs) == 3

    def test_history_is_hash_only(self, tmp_path: Path) -> None:
        # Same discipline as the audit ledger: history shows THAT it progressed and what
        # it cost, never the content.
        store = SessionStore(tmp_path)
        agent = _agent()
        agent.set_session(store, "s1")
        agent.run(_Input(task="secret task text"))
        raw = store.path("s1").read_text()
        assert "secret task text" not in raw

    def test_a_failed_run_is_recorded_as_error(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        agent = _agent(_Failing)
        agent.set_session(store, "s1")
        with pytest.raises(ValueError):
            agent.run(_Input(task="t"))
        run = store.require("s1").runs[0]
        assert run.status == "error" and "boom" in (run.error or "")


class TestBestEffort:
    def test_a_store_failure_never_fails_the_run(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        agent = _agent()
        agent.set_session(store, "s1")

        def _boom(*args: object, **kwargs: object) -> object:
            raise RuntimeError("disk on fire")

        agent._session_store.record_run = _boom  # type: ignore[method-assign, union-attr]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert agent.run(_Input(task="t")).answer == "step 1"

    def test_a_store_failure_warns(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        agent = _agent()
        agent.set_session(store, "s1")

        def _boom(*args: object, **kwargs: object) -> object:
            raise RuntimeError("disk on fire")

        agent._session_store.record_run = _boom  # type: ignore[method-assign, union-attr]
        with pytest.warns(UserWarning, match="session record failed"):
            agent.run(_Input(task="t"))
