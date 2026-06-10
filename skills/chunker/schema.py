"""Typed input/output models for ChunkerSkill."""

from __future__ import annotations

from pydantic import BaseModel

from lottie.knowledge.chunking import ChunkConfig
from lottie.knowledge.schema import Chunk, Document


class ChunkerInput(BaseModel):
    """Input for ChunkerSkill."""

    document: Document
    config: ChunkConfig = ChunkConfig()


class ChunkerOutput(BaseModel):
    """Output from ChunkerSkill."""

    chunks: list[Chunk] = []
