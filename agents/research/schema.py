"""Typed input/output models for ResearchAgent."""

from __future__ import annotations

from pydantic import BaseModel

from lottie.knowledge.schema import KnowledgeLayer


class Citation(BaseModel):
    """A source reference pointing to the exact chunk that grounded an answer."""

    doc_id: str
    chunk_id: str
    score: float
    source: str


class ResearchInput(BaseModel):
    """Input for ResearchAgent."""

    query: str
    k: int = 5
    layers: list[KnowledgeLayer] = []
    expand_graph: bool = True


class ResearchOutput(BaseModel):
    """Output from ResearchAgent."""

    digest: str
    points: list[str] = []
    citations: list[Citation] = []
