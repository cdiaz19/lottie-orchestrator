"""MemoryAgent — LLM-driven consolidation of episodic memory into semantic notes.

`MemoryAgent` reads recent episodic records via `self.memory`, asks the injected
LLM to consolidate them, and writes the resulting notes back as SEMANTIC
records. `MockMemoryAgent` prewires it with mock dependencies for tests.
"""

from __future__ import annotations

from lottie.core import BaseAgent
from lottie.llm import Message, MockLLMProvider
from lottie.memory.base import MemoryClient
from lottie.memory.mock import MockMemoryClient
from lottie.memory.schema import (
    MemoryQuery,
    MemoryRecord,
    MemoryTier,
    ReflectionInput,
    ReflectionResult,
)

REFLECT_SYSTEM_PROMPT = (
    "You consolidate an agent's recent episodic memory into durable notes. "
    "Read the log below and produce concise, standalone semantic notes — one "
    "per line, no numbering or bullets."
)


class MemoryAgent(BaseAgent[ReflectionInput, ReflectionResult]):
    """Consolidates recent episodic memory into semantic notes via the LLM."""

    def _execute(self, data: ReflectionInput) -> ReflectionResult:
        recalled = self.memory.recall(
            MemoryQuery(
                text="",
                namespace=data.namespace,
                tier=MemoryTier.EPISODIC,
                limit=data.limit,
            )
        )
        episodic = [hit.record.content for hit in recalled.hits]
        response = self.complete(
            [
                Message(role="system", content=REFLECT_SYSTEM_PROMPT),
                Message(role="user", content="\n".join(episodic)),
            ]
        )
        notes = [line.strip() for line in response.content.splitlines() if line.strip()]
        written = [
            self.memory.remember(
                MemoryRecord(
                    content=note,
                    tier=MemoryTier.SEMANTIC,
                    namespace=data.namespace,
                    tags=["reflection"],
                )
            )
            for note in notes
        ]
        return ReflectionResult(
            notes=notes,
            consolidated_count=len(episodic),
            written_ids=written,
        )


class MockMemoryAgent(MemoryAgent):
    """MemoryAgent prewired with a mock LLM + mock client for tests."""

    def __init__(
        self,
        responses: list[str] | None = None,
        memory: MemoryClient | None = None,
    ) -> None:
        super().__init__(
            llm=MockLLMProvider(responses or ["note one\nnote two"]),
            memory=memory or MockMemoryClient(),
        )
