"""Scaffold smoke tests for DocumentIngestSkill — replaced by test_ingest.py.

These tests are kept minimal to avoid duplicating the comprehensive coverage
in test_ingest.py. They verify that the public API surface matches what the
scaffold expected (schema names, importability).
"""

from __future__ import annotations

from skills.document_ingest.schema import (
    DocumentIngestInput,
    DocumentIngestOutput,
    IngestSource,
)
from skills.document_ingest.skill import DocumentIngestSkill


def test_schema_importable() -> None:
    """The public schema names are importable."""
    assert DocumentIngestInput is not None
    assert DocumentIngestOutput is not None
    assert IngestSource is not None
    assert DocumentIngestSkill is not None
