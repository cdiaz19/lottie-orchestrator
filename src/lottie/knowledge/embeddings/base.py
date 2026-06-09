"""Provider-agnostic embedding interface.

All embedding access in Lottie goes through `EmbeddingProvider`. Concrete
adapters (litellm, mock) implement `embed`; agent and skill code never imports
a vendor SDK directly.  This mirrors the pattern of `lottie.llm.LLMProvider`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from lottie.knowledge.schema import Embedding


class EmbeddingProvider(ABC):
    """Abstract embedding provider. Swap implementations via config."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Identifier of the active embedding model, e.g. 'openai/text-embedding-3-small'."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[Embedding]:
        """Embed a batch of texts and return one `Embedding` per input string."""
