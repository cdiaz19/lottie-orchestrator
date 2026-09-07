"""A plugin is only ever SUBSCRIBED — never mounted in the chain (E7).

The `isinstance(instance, Subscriber)` check at load is an early, friendly error.
The real structural guarantee is here: a plugin only ever reaches `bus.subscribe`, so even
a class shaped like middleware cannot intercept anything.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from pydantic import BaseModel

from lottie.core.base_agent import BaseAgent
from lottie.core.middleware import build_chain
from lottie.llm import MockLLMProvider
from lottie.runtime.events import RunCompleted, RunEvent, RunStarted


class _In(BaseModel):
    task: str


class _Out(BaseModel):
    answer: str


class _Agent(BaseAgent[_In, _Out]):
    def _execute(self, data: _In) -> _Out:
        return _Out(answer=data.task.upper())


class _Recorder:
    name = "recorder"

    def __init__(self) -> None:
        self.seen: list[RunEvent] = []

    def on_event(self, event: RunEvent) -> None:
        self.seen.append(event)


class _Saboteur:
    name = "saboteur"

    def on_event(self, event: RunEvent) -> None:
        raise RuntimeError("plugin sabotage")


def _agent(*plugins: object) -> _Agent:
    agent = _Agent(llm=MockLLMProvider(responses=["ok"]), enable_benchmarks=False)
    agent.set_plugins(list(plugins))  # type: ignore[arg-type]
    return agent


class TestReceivesEvents:
    def test_a_plugin_receives_real_run_events(self) -> None:
        rec = _Recorder()
        _agent(rec).run(_In(task="hi"))
        assert [type(e).__name__ for e in rec.seen] == ["RunStarted", "RunCompleted"]

    def test_several_plugins_all_receive(self) -> None:
        first, second = _Recorder(), _Recorder()
        _agent(first, second).run(_In(task="hi"))
        assert len(first.seen) == 2 and len(second.seen) == 2

    def test_no_plugins_means_no_plugin_code_runs(self) -> None:
        agent = _Agent(llm=MockLLMProvider(responses=["ok"]), enable_benchmarks=False)
        assert agent.run(_In(task="hi")).answer == "HI"


class TestNeverInTheChain:
    def test_a_plugin_is_not_mounted_as_middleware(self) -> None:
        """The structural guarantee, asserted directly.

        Even a plugin whose class looks like middleware cannot intercept, because it is
        only ever handed to `bus.subscribe` — it never enters `build_chain`.
        """
        rec = _Recorder()
        agent = _agent(rec)
        assert "recorder" not in {m.name for m in build_chain(agent)}  # type: ignore[arg-type]

    def test_adding_plugins_does_not_change_the_chain(self) -> None:
        without = {m.name for m in build_chain(_agent())}  # type: ignore[arg-type]
        with_plugin = {m.name for m in build_chain(_agent(_Recorder()))}  # type: ignore[arg-type]
        assert without == with_plugin


class TestCannotBreakARun:
    def test_a_sabotaging_plugin_cannot_fail_a_run(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert _agent(_Saboteur()).run(_In(task="hi")).answer == "HI"

    def test_a_sabotaging_plugin_cannot_starve_the_next_one(self) -> None:
        survivor = _Recorder()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _agent(_Saboteur(), survivor).run(_In(task="hi"))
        assert len(survivor.seen) == 2

    def test_a_sabotaging_plugin_cannot_displace_the_audit_record(self, tmp_path: Path) -> None:
        # Plugins subscribe AFTER the audit subscriber, and the bus isolates each
        # dispatch — so the ledger write happens regardless.
        from lottie.governance.audit import SqliteAuditLogger

        agent = _agent(_Saboteur())
        agent._audit = SqliteAuditLogger(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent.run(_In(task="hi"))
        assert len(SqliteAuditLogger(tmp_path).query()) == 1

    def test_a_sabotaging_plugin_warns_and_names_itself(self) -> None:
        with pytest.warns(UserWarning, match="saboteur"):
            _agent(_Saboteur()).run(_In(task="hi"))


class TestSeesNoRawContent:
    def test_a_plugin_cannot_read_the_task(self) -> None:
        """Events are hash-only — a plugin never sees a task, prompt, or output."""
        rec = _Recorder()
        _agent(rec).run(_In(task="SENSITIVE_TASK_TEXT"))
        assert all("SENSITIVE_TASK_TEXT" not in e.model_dump_json() for e in rec.seen)

    def test_a_plugin_cannot_read_the_output(self) -> None:
        rec = _Recorder()
        _agent(rec).run(_In(task="secret"))
        completed = [e for e in rec.seen if isinstance(e, RunCompleted)]
        assert completed and "SECRET" not in completed[0].model_dump_json()

    def test_what_a_plugin_does_get_is_hashes_and_scalars(self) -> None:
        rec = _Recorder()
        _agent(rec).run(_In(task="hi"))
        started = next(e for e in rec.seen if isinstance(e, RunStarted))
        assert len(started.input_sha256) == 64 and started.runnable == "_Agent"
