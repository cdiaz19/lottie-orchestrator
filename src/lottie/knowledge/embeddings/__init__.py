"""Embedding provider abstraction for the Lottie knowledge subsystem.

All embedding access goes through `EmbeddingProvider`.  Concrete adapters
(litellm, mock) implement the interface; agent and skill code never imports a
vendor SDK directly.

Exported names
--------------
EmbeddingProvider         — the ABC (import and subclass to add a real adapter)
MockEmbeddingProvider     — deterministic in-memory provider for tests
LiteLLMEmbeddingProvider  — litellm-backed provider for real embeddings
build_embedding_provider  — factory: returns Mock or LiteLLM based on model prefix
"""

from __future__ import annotations

from lottie.knowledge.embeddings.base import EmbeddingProvider
from lottie.knowledge.embeddings.litellm_provider import LiteLLMEmbeddingProvider
from lottie.knowledge.embeddings.mock import MockEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "MockEmbeddingProvider",
    "LiteLLMEmbeddingProvider",
    "build_embedding_provider",
]


def build_embedding_provider(model: str) -> EmbeddingProvider:
    """Return the appropriate ``EmbeddingProvider`` for *model*.

    Dispatch rules
    --------------
    - ``model == "mock"`` or ``model.startswith("mock/")``  → :class:`MockEmbeddingProvider`
    - anything else                                          → :class:`LiteLLMEmbeddingProvider`

    Examples::

        build_embedding_provider("mock/embed")                    # MockEmbeddingProvider
        build_embedding_provider("mock")                          # MockEmbeddingProvider
        build_embedding_provider("openai/text-embedding-3-small") # LiteLLMEmbeddingProvider
        build_embedding_provider("cohere/embed-english-v3.0")     # LiteLLMEmbeddingProvider
    """
    if model == "mock" or model.startswith("mock/"):
        return MockEmbeddingProvider(model=model)
    return LiteLLMEmbeddingProvider(model=model)
