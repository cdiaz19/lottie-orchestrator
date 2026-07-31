"""End-to-end: runs produce episodic records, and MemoryAgent consolidates them.

Before S3a this path could not be exercised — `MemoryAgent._execute` read an
always-empty EPISODIC tier, so `lottie reflect` was a no-op in any real project.
"""

from __future__ import annotations

from pydantic import BaseModel

from lottie.core.base_agent import BaseAgent
from lottie.llm import MockLLMProvider
from lottie.memory.agent import MemoryAgent
from lottie.memory.mock import MockMemoryClient
from lottie.memory.schema import MemoryQuery, MemoryTier, ReflectionInput


class _Input(BaseModel):
    task: str


class _Output(BaseModel):
    answer: str


class _Worker(BaseAgent[_Input, _Output]):
    def _execute(self, data: _Input) -> _Output:
        return _Output(answer=data.task.upper())


def _worker(memory: MockMemoryClient) -> _Worker:
    agent = _Worker(
        llm=MockLLMProvider(responses=["ok"]), memory=memory, enable_benchmarks=False
    )
    agent.set_trajectory(enabled=True, namespace="ns", max_chars=4000)
    return agent


def _tier(memory: MockMemoryClient, tier: MemoryTier) -> list[str]:
    query = MemoryQuery(text="", namespace="ns", tier=tier, limit=100)
    return [hit.record.content for hit in memory.recall(query).hits]


def test_runs_populate_the_episodic_tier() -> None:
    memory = MockMemoryClient()
    worker = _worker(memory)
    for task in ("alpha", "beta", "gamma"):
        worker.run(_Input(task=task))
    assert len(_tier(memory, MemoryTier.EPISODIC)) == 3


def test_memory_agent_consolidates_what_the_runs_wrote() -> None:
    memory = MockMemoryClient()
    worker = _worker(memory)
    for task in ("alpha", "beta"):
        worker.run(_Input(task=task))

    consolidator = MemoryAgent(
        llm=MockLLMProvider(responses=["- uppercasing is the common pattern"]),
        memory=memory,
    )
    result = consolidator.run(ReflectionInput(namespace="ns", limit=50))
    # The number that was structurally always zero before this slice.
    assert result.consolidated_count == 2


def test_consolidation_writes_semantic_notes_back() -> None:
    memory = MockMemoryClient()
    worker = _worker(memory)
    worker.run(_Input(task="alpha"))

    consolidator = MemoryAgent(
        llm=MockLLMProvider(responses=["- uppercasing is the common pattern"]),
        memory=memory,
    )
    consolidator.run(ReflectionInput(namespace="ns", limit=50))
    assert _tier(memory, MemoryTier.SEMANTIC) != []
