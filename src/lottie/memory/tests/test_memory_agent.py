from __future__ import annotations

from lottie.memory.agent import MemoryAgent, MockMemoryAgent
from lottie.memory.mock import MockMemoryClient
from lottie.memory.schema import MemoryRecord, MemoryTier, ReflectionInput


def _episodic(content: str) -> MemoryRecord:
    return MemoryRecord(content=content, namespace="demo", tier=MemoryTier.EPISODIC)


def test_mock_memory_agent_consolidates_and_writes_back() -> None:
    client = MockMemoryClient(records=[_episodic("user asked X"), _episodic("user asked Y")])
    agent = MockMemoryAgent(responses=["note one\nnote two"], memory=client)

    result = agent.run(ReflectionInput(namespace="demo"))

    assert result.notes == ["note one", "note two"]
    assert result.consolidated_count == 2
    assert len(result.written_ids) == 2
    # Two new SEMANTIC records written back, tagged reflection.
    semantic = [r for r in client.records if r.tier is MemoryTier.SEMANTIC]
    assert [r.content for r in semantic] == ["note one", "note two"]
    assert all(r.tags == ["reflection"] for r in semantic)


def test_mock_memory_agent_blank_lines_ignored() -> None:
    client = MockMemoryClient(records=[_episodic("e")])
    agent = MockMemoryAgent(responses=["  alpha  \n\n   \nbeta\n"], memory=client)
    result = agent.run(ReflectionInput(namespace="demo"))
    assert result.notes == ["alpha", "beta"]


def test_mock_memory_agent_defaults_are_usable() -> None:
    # No args: canned response + fresh empty client. Nothing to consolidate.
    agent = MockMemoryAgent()
    result = agent.run(ReflectionInput(namespace="demo"))
    assert result.consolidated_count == 0
    assert result.notes == ["note one", "note two"]


def test_memory_agent_is_base_agent_subclass() -> None:
    from lottie.core import BaseAgent

    assert issubclass(MemoryAgent, BaseAgent)
