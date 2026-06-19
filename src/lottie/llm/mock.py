"""In-memory `LLMProvider` for tests.

Returns pre-defined responses in order and records the messages it received,
so agent integration tests can assert on decision logic without a real LLM.
Unit tests must never call a real provider (CLAUDE.md rule 5).
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping

from lottie.llm.base import LLMProvider, LLMResponse, Message, TokenUsage


class MockLLMProvider(LLMProvider):
    """Deterministic provider that replays a fixed list of responses."""

    def __init__(self, responses: list[str], model: str = "mock/mock-model") -> None:
        if not responses:
            raise ValueError("MockLLMProvider needs at least one response")
        self._responses = responses
        self._model = model
        self._index = 0
        self.calls: list[list[Message]] = []

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> LLMResponse:
        self.calls.append(messages)
        if self._index >= len(self._responses):
            raise RuntimeError("MockLLMProvider responses exhausted")
        content = self._responses[self._index]
        self._index += 1
        return LLMResponse(content=content, usage=TokenUsage(), model=self._model)

    def stream(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> Iterator[str]:
        """Replay the next canned response as multiple deltas (so tests see real chunking).

        Consumes the same response queue as `complete` (records the call, advances the index), then
        yields pieces that reconstruct the response EXACTLY (`"".join(deltas) == response`)."""
        self.calls.append(messages)
        if self._index >= len(self._responses):
            raise RuntimeError("MockLLMProvider responses exhausted")
        content = self._responses[self._index]
        self._index += 1
        # non-space-run+trailing-space and pure-space runs both captured -> join == content.
        yield from re.findall(r"\S+\s*|\s+", content)
