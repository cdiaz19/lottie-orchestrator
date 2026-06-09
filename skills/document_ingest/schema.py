"""Typed input/output models for DocumentIngestSkill."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from lottie.knowledge.chunking import ChunkConfig
from lottie.knowledge.schema import Document, KnowledgeLayer


class IngestSource(BaseModel):
    """A single source of content to ingest into the knowledge layer.

    Parameters
    ----------
    kind:
        ``"text"`` — raw string content; ``"file"`` — path on disk;
        ``"url"`` — remote URL (deferred, raises ``NotImplementedError``).
    value:
        The source payload: the raw text, a file path, or a URL.
    layer:
        Requested *eventual* layer after human promotion. Phase 1 always
        writes to ``KnowledgeLayer.DRAFT`` regardless of this value.
    """

    kind: Literal["file", "text", "url"]
    value: str
    layer: KnowledgeLayer = KnowledgeLayer.DRAFT


class DocumentIngestInput(BaseModel):
    """Input for DocumentIngestSkill."""

    sources: list[IngestSource]
    config: ChunkConfig = ChunkConfig()


class DocumentIngestOutput(BaseModel):
    """Output from DocumentIngestSkill."""

    documents: list[Document] = []
    chunk_count: int = 0
    flagged: list[str] = []
    """Identifiers of sources rejected by the security gate."""
