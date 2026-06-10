"""DocumentIngestSkill — reference skill entry point.

The canonical implementation lives in ``lottie.knowledge.ingest`` (framework).
This module is a thin re-export so that the skill remains discoverable via
``lottie list skills`` and existing imports from ``skills.document_ingest.skill``
continue to work unchanged.
"""

from __future__ import annotations

from lottie.knowledge.ingest import DocumentIngestSkill  # noqa: F401

__all__ = ["DocumentIngestSkill"]
