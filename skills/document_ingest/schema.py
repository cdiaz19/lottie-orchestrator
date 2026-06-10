"""Typed input/output models for DocumentIngestSkill.

The canonical home of these models is ``lottie.knowledge.ingest`` (framework).
This module is a thin re-export so that code importing from
``skills.document_ingest.schema`` continues to work unchanged.
"""

from __future__ import annotations

from lottie.knowledge.ingest import (  # noqa: F401
    DocumentIngestInput,
    DocumentIngestOutput,
    IngestSource,
)

__all__ = [
    "DocumentIngestInput",
    "DocumentIngestOutput",
    "IngestSource",
]
