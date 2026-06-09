"""Embedding provider abstraction for the Lottie knowledge subsystem.

All embedding access goes through `EmbeddingProvider`.  Concrete adapters
(litellm, mock) implement the interface; agent and skill code never imports a
vendor SDK directly.

Exported names
--------------
EmbeddingProvider  — the ABC (import and subclass to add a real adapter)
MockEmbeddingProvider — deterministic in-memory provider for tests
"""

from __future__ import annotations

from lottie.knowledge.embeddings.base import EmbeddingProvider
from lottie.knowledge.embeddings.mock import MockEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "MockEmbeddingProvider",
]
