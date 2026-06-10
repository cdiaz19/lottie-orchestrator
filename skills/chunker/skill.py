"""ChunkerSkill — deterministic recursive-character chunker.

Delegates to ``chunk_document`` in ``lottie.knowledge.chunking``; no LLM
is involved. Suitable for unit testing without mocks.
"""

from __future__ import annotations

from lottie.core import BaseSkill
from lottie.knowledge.chunking import chunk_document

from .schema import ChunkerInput, ChunkerOutput


class ChunkerSkill(BaseSkill[ChunkerInput, ChunkerOutput]):
    """Split a :class:`~lottie.knowledge.schema.Document` into ordered
    :class:`~lottie.knowledge.schema.Chunk` objects with character offsets.

    Purely deterministic — same input always produces the same output.
    No LLM, no external calls, no side effects.
    """

    def _execute(self, data: ChunkerInput) -> ChunkerOutput:
        return ChunkerOutput(chunks=chunk_document(data.document, data.config))
