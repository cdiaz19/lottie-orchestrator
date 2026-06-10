"""Pydantic models for the knowledge subsystem.

Pure data shapes — no logic, no imports beyond pydantic/stdlib. Loaders,
retrievers, and graph builders all build on these.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, model_validator


class KnowledgeLayer(StrEnum):
    """The five knowledge layers."""

    GLOBAL = "global"      # always-injected global context
    PLATFORM = "platform"  # per-platform context
    PROJECT = "project"    # per-project context
    MEMORY = "memory"      # task-relevant, tag-matched
    DRAFT = "draft"        # AI-generated, awaiting human review


class DocStatus(StrEnum):
    """Lifecycle status of a knowledge document."""

    DRAFT = "draft"      # document lifecycle — distinct from KnowledgeLayer.DRAFT (folder)
    CURATED = "curated"
    AGING = "aging"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class Document(BaseModel):
    """A knowledge document loaded from a YAML-frontmatter file."""

    id: str
    source: str
    layer: KnowledgeLayer
    content: str
    frontmatter: dict[str, str] = {}
    tags: list[str] = []
    depends_on: list[str] = []
    status: DocStatus = DocStatus.DRAFT


class Chunk(BaseModel):
    """A contiguous slice of a Document's content."""

    id: str
    doc_id: str
    index: int
    text: str
    start: int
    end: int
    metadata: dict[str, str] = {}


class Embedding(BaseModel):
    """A dense vector embedding for a chunk of text."""

    vector: list[float]
    model: str
    dim: int

    @model_validator(mode="after")
    def _check_dim_matches_vector(self) -> Embedding:
        if len(self.vector) != self.dim:
            raise ValueError(
                f"Embedding.dim={self.dim} but vector has {len(self.vector)} elements"
            )
        return self


class EmbeddedChunk(BaseModel):
    """A Chunk paired with its Embedding."""

    chunk: Chunk
    embedding: Embedding


class RetrievalQuery(BaseModel):
    """A retrieval request against the knowledge store."""

    text: str
    k: int = 5
    layers: list[KnowledgeLayer] = []
    tags: list[str] = []
    expand_graph: bool = False


class RetrievalHit(BaseModel):
    """One retrieved chunk with its relevance score."""

    chunk: Chunk
    score: float


class RetrievalResult(BaseModel):
    """Ordered hits for a retrieval query."""

    hits: list[RetrievalHit] = []
