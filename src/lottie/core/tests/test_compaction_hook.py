"""Compaction wired into BaseAgent.complete — the single call site (V3 spec §1.1)."""

from __future__ import annotations

import warnings

import pytest
from pydantic import BaseModel

from lottie.core.base_agent import BaseAgent
from lottie.governance.cost import TokenCapExceeded
from lottie.llm import Message, MockLLMProvider
from lottie.memory.compaction import SUMMARY_PREFIX


class _Input(BaseModel):
    task: str


class _Output(BaseModel):
    answer: str


class _Chatty(BaseAgent[_Input, _Output]):
    """Sends a long history on every completion."""

    turns: int = 12

    def _execute(self, data: _Input) -> _Output:
        history = [
            Message(role="user", content=f"turn {i} " + "x" * 400) for i in range(self.turns)
        ]
        response = self.complete(history)
        return _Output(answer=response.content)


def _agent(
    *, enabled: bool = True, max_tokens: int = 200, keep_recent: int = 2, responses: int = 6
) -> _Chatty:
    agent = _Chatty(
        llm=MockLLMProvider(responses=["summary text"] * responses),
        enable_benchmarks=False,
    )
    agent.set_compaction(
        enabled=enabled, max_context_tokens=max_tokens, keep_recent=keep_recent
    )
    return agent


def _sent(agent: _Chatty) -> list[Message]:
    """The messages the provider actually received on the last call."""
    # The summariser also goes through llm.complete, so the LAST call is the real
    # completion — which is exactly the prompt compaction produced.
    calls = agent.llm.calls  # type: ignore[attr-defined]
    return list(calls[-1])


class TestOptIn:
    def test_off_by_default(self) -> None:
        agent = _Chatty(llm=MockLLMProvider(responses=["ok"]), enable_benchmarks=False)
        agent.run(_Input(task="t"))
        assert len(_sent(agent)) == 12

    def test_disabled_sends_everything(self) -> None:
        agent = _agent(enabled=False)
        agent.run(_Input(task="t"))
        assert len(_sent(agent)) == 12


class TestCompactionApplies:
    def test_a_long_context_is_compacted(self) -> None:
        agent = _agent()
        agent.run(_Input(task="t"))
        assert len(_sent(agent)) < 12

    def test_the_compacted_prompt_carries_a_summary(self) -> None:
        agent = _agent()
        agent.run(_Input(task="t"))
        assert any(m.content.startswith(SUMMARY_PREFIX) for m in _sent(agent))

    def test_recent_turns_survive_verbatim(self) -> None:
        agent = _agent(keep_recent=2)
        agent.run(_Input(task="t"))
        assert _sent(agent)[-1].content.startswith("turn 11")

    def test_a_short_context_is_untouched_and_costs_no_llm_call(self) -> None:
        agent = _agent(max_tokens=1_000_000)
        agent.run(_Input(task="t"))
        # 12 history messages, no summary inserted, and only the one real completion.
        assert len(_sent(agent)) == 12
        assert not any(m.content.startswith(SUMMARY_PREFIX) for m in _sent(agent))


class TestRecursionSafety:
    def test_summarising_does_not_re_enter_compaction(self) -> None:
        """`_summarize_span` must call self.llm.complete, never self.complete.

        If it re-entered, this run would recurse until the mock ran out of responses
        (or the stack blew), rather than completing.
        """
        agent = _agent(responses=4)
        assert agent.run(_Input(task="t")).answer == "summary text"


class TestRecallIsPinned:
    """Recall-as-data is a security contract (S2a); compacting it away would weaken it
    silently. BaseAgent pins system messages for exactly this reason.

    Driven through `complete()` rather than `run()`: `run` calls `_load_recall`, which
    resets the prefix when recall is disabled, so a prefix set by hand would be wiped
    before the code under test ever saw it.
    """

    def test_the_recall_block_survives_compaction(self) -> None:
        agent = _agent()
        agent._recall_prefix = "<recalled-notes>" + "y" * 400 + "</recalled-notes>"
        history = [Message(role="user", content=f"turn {i} " + "x" * 400) for i in range(12)]
        agent.complete(history)
        assert any("recalled-notes" in m.content for m in _sent(agent))

    def test_the_history_around_it_is_still_compacted(self) -> None:
        agent = _agent()
        agent._recall_prefix = "<recalled-notes>" + "y" * 400 + "</recalled-notes>"
        history = [Message(role="user", content=f"turn {i} " + "x" * 400) for i in range(12)]
        agent.complete(history)
        assert len(_sent(agent)) < 13  # 12 turns + the pinned recall block


class TestBestEffort:
    def test_a_summariser_failure_sends_the_full_context(self) -> None:
        agent = _agent()

        def _boom(messages: list[Message]) -> str:
            raise RuntimeError("summariser down")

        agent._summarize_span = _boom  # type: ignore[method-assign]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent.run(_Input(task="t"))
        assert len(_sent(agent)) == 12

    def test_a_summariser_failure_warns(self) -> None:
        agent = _agent()

        def _boom(messages: list[Message]) -> str:
            raise RuntimeError("summariser down")

        agent._summarize_span = _boom  # type: ignore[method-assign]
        with pytest.warns(UserWarning, match="compaction failed"):
            agent.run(_Input(task="t"))

    def test_a_budget_stop_is_not_swallowed(self) -> None:
        # A token-cap trip is the run's decision; compaction must not convert it into a
        # warning and carry on spending.
        agent = _agent()

        def _cap(messages: list[Message]) -> str:
            raise TokenCapExceeded("cap")

        agent._summarize_span = _cap  # type: ignore[method-assign]
        with pytest.raises(TokenCapExceeded):
            agent.run(_Input(task="t"))


class TestKeepRecentFloor:
    def test_keep_recent_is_floored_at_one(self) -> None:
        # keep_recent=0 would make the task itself droppable.
        agent = _agent(keep_recent=0)
        assert agent._keep_recent == 1
