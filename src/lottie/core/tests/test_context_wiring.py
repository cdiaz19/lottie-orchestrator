"""The compiler wired into `complete()` (E4 S1).

The behavioural claim is that nothing changes for existing agents: `complete(messages)`
keeps its signature, and a prompt with only pinned sources comes out byte-identical.
"""

from __future__ import annotations

import warnings

import pytest
from pydantic import BaseModel

from lottie.context.compiler import StaticSource
from lottie.core.base_agent import BaseAgent
from lottie.llm import Message, MockLLMProvider


class _In(BaseModel):
    task: str


class _Out(BaseModel):
    answer: str


class _Agent(BaseAgent[_In, _Out]):
    def _execute(self, data: _In) -> _Out:
        response = self.complete(
            [
                Message(role="system", content="You are helpful."),
                Message(role="user", content=data.task),
            ]
        )
        return _Out(answer=response.content)


def _agent() -> _Agent:
    return _Agent(llm=MockLLMProvider(responses=["ok"] * 4), enable_benchmarks=False)


def _sent(agent: _Agent) -> list[Message]:
    return list(agent.llm.calls[-1])  # type: ignore[attr-defined]


class TestByteIdenticalWithoutRecall:
    def test_the_prompt_is_unchanged(self) -> None:
        agent = _agent()
        agent.run(_In(task="hello"))
        assert [m.content for m in _sent(agent)] == ["You are helpful.", "hello"]

    def test_message_order_is_preserved(self) -> None:
        agent = _agent()
        agent.run(_In(task="hello"))
        assert _sent(agent)[0].role == "system" and _sent(agent)[-1].role == "user"


class TestRecallIsAPinnedSource:
    def test_recall_is_prepended_ahead_of_the_agent_messages(self) -> None:
        # Order 20 (recall) before order 90 (agent) — the assembly authority, not a
        # hardcoded prepend.
        agent = _agent()
        agent._recall_prefix = "<recalled-notes>note</recalled-notes>"
        agent.complete([Message(role="user", content="task")])
        assert "recalled-notes" in _sent(agent)[0].content

    def test_recall_is_declared_pinned(self) -> None:
        # S2a's anti-poisoning contract is a SOURCE property now, not a role check.
        agent = _agent()
        agent._recall_prefix = "<recalled-notes>note</recalled-notes>"
        sources = {s.name: s.pinned for s in agent._context_sources([])}
        assert sources["recall"] is True

    def test_the_agent_messages_are_pinned(self) -> None:
        # Dropping the task is never the right trade.
        agent = _agent()
        sources = {s.name: s.pinned for s in agent._context_sources([])}
        assert sources["agent"] is True

    def test_no_recall_means_no_recall_source(self) -> None:
        agent = _agent()
        assert "recall" not in {s.name for s in agent._context_sources([])}


class TestBestEffort:
    def test_an_assembly_failure_sends_the_prompt_as_is(self) -> None:
        agent = _agent()

        def _boom(messages: list[Message]) -> list[StaticSource]:
            raise RuntimeError("assembly down")

        agent._context_sources = _boom  # type: ignore[method-assign]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent.run(_In(task="hello"))
        assert [m.content for m in _sent(agent)] == ["You are helpful.", "hello"]

    def test_an_assembly_failure_warns(self) -> None:
        agent = _agent()

        def _boom(messages: list[Message]) -> list[StaticSource]:
            raise RuntimeError("assembly down")

        agent._context_sources = _boom  # type: ignore[method-assign]
        with pytest.warns(UserWarning, match="context assembly failed"):
            agent.run(_In(task="hello"))
