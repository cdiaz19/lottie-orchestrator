"""In-memory `EmbeddingProvider` for tests.

Produces deterministic L2-normalised vectors from SHA-256 hashes of the input
text.  No API key, no network, no vendor SDK — unit tests must never call a
real provider (CLAUDE.md rule 5).

Arbitrary ``dim`` values are supported.  A single SHA-256 digest is 32 bytes;
for ``dim > 32`` the implementation generates additional bytes by iteratively
re-hashing: ``sha256(digest + counter.to_bytes(2, "big"))`` for counter = 1, 2, …
Each byte is mapped to a float in [-1, 1] via ``(byte / 255.0) * 2 - 1``.
The resulting vector is L2-normalised before being returned.
"""

from __future__ import annotations

import hashlib

from lottie.knowledge.embeddings.base import EmbeddingProvider
from lottie.knowledge.schema import Embedding


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider that derives vectors from SHA-256 hashes."""

    def __init__(self, *, dim: int = 16, model: str = "mock/embed") -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self._dim = dim
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        """Dimensionality of the produced vectors."""
        return self._dim

    def embed(self, texts: list[str]) -> list[Embedding]:
        """Embed each text deterministically and return L2-normalised `Embedding` objects."""
        return [self._embed_one(text) for text in texts]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _raw_bytes(self, text: str) -> bytes:
        """Generate exactly ``self._dim`` deterministic bytes from *text*.

        Strategy:
        - Round 0: ``sha256(text.encode("utf-8"))`` → 32 bytes.
        - Round k (k >= 1): ``sha256(round_0_digest + bytes([k]))`` → 32 more bytes.
        Bytes are concatenated and the first ``self._dim`` are returned.
        """
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        buf = bytearray(seed)
        counter = 1
        while len(buf) < self._dim:
            buf.extend(hashlib.sha256(seed + counter.to_bytes(2, "big")).digest())
            counter += 1
        return bytes(buf[: self._dim])

    def _embed_one(self, text: str) -> Embedding:
        raw = self._raw_bytes(text)
        vec = [(b / 255.0) * 2.0 - 1.0 for b in raw]

        # L2-normalise
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0.0:
            vec = [x / norm for x in vec]

        return Embedding(vector=vec, model=self._model, dim=self._dim)
