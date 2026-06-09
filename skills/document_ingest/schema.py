"""Typed input/output models for DocumentIngestSkill."""

from __future__ import annotations

from pydantic import BaseModel

from lottie.knowledge.chunking import ChunkConfig
from lottie.knowledge.ingest import IngestSource as IngestSource  # re-export
from lottie.knowledge.schema import Document  # noqa: F401
from lottie.knowledge.schema import KnowledgeLayer as KnowledgeLayer


class DocumentIngestInput(BaseModel):
    """Input for DocumentIngestSkill."""

    sources: list[IngestSource]
    config: ChunkConfig = ChunkConfig()


class DocumentIngestOutput(BaseModel):
    """Output from DocumentIngestSkill."""

    documents: list[Document] = []
    chunk_count: int = 0
    flagged: list[str] = []
    """Draft IDs (with ``draft/`` prefix) of sources rejected by the security gate."""
    errors: list[str] = []
    """Load/processing failures (bad path, URL not implemented, empty content, etc.).
    Each entry is a string of the form ``"<source identifier>: <reason>"``.
    These are distinct from ``flagged`` (security rejections) — no security gate was
    reached for errored sources.
    """
