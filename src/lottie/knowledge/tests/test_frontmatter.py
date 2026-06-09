"""Tests for lottie.knowledge.frontmatter — YAML frontmatter parser."""

from __future__ import annotations

from pathlib import Path

from lottie.knowledge.frontmatter import parse_frontmatter, to_document
from lottie.knowledge.schema import DocStatus, KnowledgeLayer

# ---------------------------------------------------------------------------
# Inline fixture — same content as tests/fixtures/knowledge/global/sample.md
# ---------------------------------------------------------------------------
FIXTURE_TEXT = """\
---
id: lottie/auth-conventions
layer: platform
scope: lottie
topic: authentication
tags: [auth, jwt, sessions]
status: curated
last_verified: 2026-05
depends_on: [global/conventions]
supersedes: []
---
Use JWT for stateless auth. Sessions expire after 24h.
"""


# ---------------------------------------------------------------------------
# parse_frontmatter tests
# ---------------------------------------------------------------------------


def test_parse_frontmatter_reads_id() -> None:
    meta, _ = parse_frontmatter(FIXTURE_TEXT)
    assert meta["id"] == "lottie/auth-conventions"


def test_parse_frontmatter_reads_tags() -> None:
    meta, _ = parse_frontmatter(FIXTURE_TEXT)
    assert meta["tags"] == ["auth", "jwt", "sessions"]


def test_parse_frontmatter_body_has_no_fence_or_id_key() -> None:
    _, body = parse_frontmatter(FIXTURE_TEXT)
    assert "---" not in body
    assert "id:" not in body
    assert "Use JWT" in body


def test_parse_frontmatter_no_frontmatter() -> None:
    text = "no frontmatter here"
    meta, body = parse_frontmatter(text)
    assert meta == {}
    assert body == text


def test_parse_frontmatter_malformed_yaml_no_raise() -> None:
    # Unbalanced braces — invalid YAML
    malformed = "---\n{bad: [yaml: nope\n---\nbody text\n"
    meta, _ = parse_frontmatter(malformed)
    assert meta == {}
    # Must not raise — the test reaching here is the assertion


def test_parse_frontmatter_empty_fence_returns_empty_meta() -> None:
    text = "---\n---\nbody here\n"
    meta, body = parse_frontmatter(text)
    assert meta == {}
    assert "body here" in body


def test_parse_frontmatter_non_dict_yaml_treated_as_no_meta() -> None:
    # yaml.safe_load of a list → treat as no metadata
    text = "---\n- item1\n- item2\n---\nbody\n"
    meta, body = parse_frontmatter(text)
    assert meta == {}
    assert "body" in body


# ---------------------------------------------------------------------------
# to_document tests
# ---------------------------------------------------------------------------


def test_to_document_id_from_frontmatter() -> None:
    path = Path("knowledge/global/auth.md")
    doc = to_document(path, KnowledgeLayer.GLOBAL, FIXTURE_TEXT)
    assert doc.id == "lottie/auth-conventions"


def test_to_document_tags() -> None:
    path = Path("knowledge/global/auth.md")
    doc = to_document(path, KnowledgeLayer.GLOBAL, FIXTURE_TEXT)
    assert doc.tags == ["auth", "jwt", "sessions"]


def test_to_document_depends_on() -> None:
    path = Path("knowledge/global/auth.md")
    doc = to_document(path, KnowledgeLayer.GLOBAL, FIXTURE_TEXT)
    assert doc.depends_on == ["global/conventions"]


def test_to_document_status_curated() -> None:
    path = Path("knowledge/global/auth.md")
    doc = to_document(path, KnowledgeLayer.GLOBAL, FIXTURE_TEXT)
    assert doc.status == DocStatus.CURATED


def test_to_document_layer_passed_through() -> None:
    path = Path("knowledge/global/auth.md")
    doc = to_document(path, KnowledgeLayer.GLOBAL, FIXTURE_TEXT)
    assert doc.layer == KnowledgeLayer.GLOBAL


def test_to_document_source_is_str_path() -> None:
    path = Path("knowledge/global/auth.md")
    doc = to_document(path, KnowledgeLayer.GLOBAL, FIXTURE_TEXT)
    assert doc.source == str(path)


def test_to_document_no_frontmatter_uses_stem_as_id() -> None:
    path = Path("knowledge/global/readme.md")
    doc = to_document(path, KnowledgeLayer.GLOBAL, "just plain text")
    assert doc.id == "readme"


def test_to_document_no_frontmatter_status_draft() -> None:
    path = Path("knowledge/global/readme.md")
    doc = to_document(path, KnowledgeLayer.GLOBAL, "just plain text")
    assert doc.status == DocStatus.DRAFT


def test_to_document_no_frontmatter_empty_tags() -> None:
    path = Path("knowledge/global/readme.md")
    doc = to_document(path, KnowledgeLayer.GLOBAL, "just plain text")
    assert doc.tags == []


def test_to_document_no_frontmatter_empty_depends_on() -> None:
    path = Path("knowledge/global/readme.md")
    doc = to_document(path, KnowledgeLayer.GLOBAL, "just plain text")
    assert doc.depends_on == []


def test_to_document_invalid_status_falls_back_to_draft() -> None:
    text = "---\nstatus: unknown-value\n---\nbody\n"
    path = Path("knowledge/global/foo.md")
    doc = to_document(path, KnowledgeLayer.GLOBAL, text)
    assert doc.status == DocStatus.DRAFT


def test_to_document_frontmatter_dict_contains_all_raw_keys() -> None:
    path = Path("knowledge/global/auth.md")
    doc = to_document(path, KnowledgeLayer.GLOBAL, FIXTURE_TEXT)
    # All top-level meta keys should be present, values coerced to str
    assert "id" in doc.frontmatter
    assert doc.frontmatter["id"] == "lottie/auth-conventions"
    assert "layer" in doc.frontmatter
    assert "tags" in doc.frontmatter
