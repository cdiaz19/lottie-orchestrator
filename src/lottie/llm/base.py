"""Provider-agnostic LLM interface.

All LLM access in Lottie goes through `LLMProvider`. Concrete adapters
(litellm, mock) implement `complete`; agent and skill code never imports a
vendor SDK directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    """A single chat message in a provider-neutral shape."""

    role: Role
    content: str


class TokenUsage(BaseModel):
    """Token counts for one LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0


class LLMResponse(BaseModel):
    """Normalized result of an LLM completion."""

    content: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    model: str
    cost_usd: float = 0.0


class LLMProvider(ABC):
    """Abstract LLM provider. Swap implementations via config."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Identifier of the active model, e.g. 'anthropic/claude-sonnet-4-6'."""

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> LLMResponse:
        """Run a completion and return a normalized `LLMResponse`."""

    def stream(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> Iterator[str]:
        """Yield assistant content deltas as they are produced.

        Default: a one-shot fallback that yields the whole `complete()` content in a single delta,
        so a provider implementing only `complete` still satisfies the streaming interface (no
        incremental latency). Real providers override this with true incremental streaming.
        """
        yield self.complete(messages, model_params).content
