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

    def _pop_response(self, messages: list[Message]) -> str:
        """Record the call and return the next canned response, advancing the queue.

        Eager (not deferred): both `complete` and `stream` call this at invocation time, so call
        recording + exhaustion behave identically across the two."""
        self.calls.append(messages)
        if self._index >= len(self._responses):
            raise RuntimeError("MockLLMProvider responses exhausted")
        content = self._responses[self._index]
        self._index += 1
        return content

    def complete(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> LLMResponse:
        content = self._pop_response(messages)
        return LLMResponse(content=content, usage=TokenUsage(), model=self._model)

    def stream(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> Iterator[str]:
        """Replay the next canned response as multiple deltas reconstructing it exactly.

        Consumes the same queue as `complete` (eagerly, via `_pop_response`)."""
        content = self._pop_response(messages)
        return (piece for piece in re.findall(r"\S+\s*|\s+", content))
