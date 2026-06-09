"""Tests for lottie.knowledge.manifest — KnowledgeManifest loader."""

from __future__ import annotations

from pathlib import Path

from lottie.knowledge.manifest import KnowledgeManifest
from lottie.knowledge.schema import KnowledgeLayer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Path layout: src/lottie/knowledge/tests/test_manifest.py
# parents[0] = tests/
# parents[1] = knowledge/
# parents[2] = lottie/
# parents[3] = src/
# parents[4] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[4]
_FIXTURES_ROOT = _REPO_ROOT / "tests" / "fixtures"


# ---------------------------------------------------------------------------
# Fixtures-based tests (real on-disk knowledge tree)
# ---------------------------------------------------------------------------


def test_from_root_documents_non_empty() -> None:
    """Loading from the fixtures root yields at least one document."""
    m = KnowledgeManifest.from_root(_FIXTURES_ROOT)
    assert len(m.documents) > 0


def test_from_root_by_id_known_doc() -> None:
    """by_id returns the sample fixture document by its frontmatter id."""
    m = KnowledgeManifest.from_root(_FIXTURES_ROOT)
    doc = m.by_id("lottie/auth-conventions")
    assert doc is not None
    assert doc.layer == KnowledgeLayer.GLOBAL


def test_from_root_by_layer_global_contains_sample() -> None:
    """by_layer(GLOBAL) contains the sample fixture document."""
    m = KnowledgeManifest.from_root(_FIXTURES_ROOT)
    global_docs = m.by_layer(KnowledgeLayer.GLOBAL)
    ids = [d.id for d in global_docs]
    assert "lottie/auth-conventions" in ids


def test_from_root_by_id_unknown_returns_none() -> None:
    """by_id with an unknown id returns None, not an error."""
    m = KnowledgeManifest.from_root(_FIXTURES_ROOT)
    assert m.by_id("nope") is None


def test_from_root_by_layer_memory_empty() -> None:
    """by_layer for a layer with no fixture files returns an empty list."""
    m = KnowledgeManifest.from_root(_FIXTURES_ROOT)
    assert m.by_layer(KnowledgeLayer.MEMORY) == []


# ---------------------------------------------------------------------------
# Empty-dir / missing knowledge/ subdir — no error
# ---------------------------------------------------------------------------


def test_from_root_missing_knowledge_dir_returns_empty(tmp_path: Path) -> None:
    """from_root on a dir with no knowledge/ subdir returns empty documents."""
    m = KnowledgeManifest.from_root(tmp_path)
    assert m.documents == []


# ---------------------------------------------------------------------------
# Tmp-tree: draft layer
# ---------------------------------------------------------------------------


def test_from_root_tmp_draft_layer(tmp_path: Path) -> None:
    """A tmp knowledge tree with a draft/foo.md is loaded with layer DRAFT."""
    draft_dir = tmp_path / "knowledge" / "draft"
    draft_dir.mkdir(parents=True)
    (draft_dir / "foo.md").write_text("---\nid: draft/foo\n---\nbody\n", encoding="utf-8")

    m = KnowledgeManifest.from_root(tmp_path)
    doc = m.by_id("draft/foo")
    assert doc is not None
    assert doc.layer == KnowledgeLayer.DRAFT


# ---------------------------------------------------------------------------
# Determinism / sorting
# ---------------------------------------------------------------------------


def test_from_root_documents_sorted_by_id(tmp_path: Path) -> None:
    """documents list is sorted by id for deterministic ordering."""
    global_dir = tmp_path / "knowledge" / "global"
    global_dir.mkdir(parents=True)
    # Write in reverse alphabetical order to prove sorting is applied
    (global_dir / "zzz.md").write_text("---\nid: z/doc\n---\nbody\n", encoding="utf-8")
    (global_dir / "aaa.md").write_text("---\nid: a/doc\n---\nbody\n", encoding="utf-8")

    m = KnowledgeManifest.from_root(tmp_path)
    assert len(m.documents) == 2
    assert m.documents[0].id == "a/doc"
    assert m.documents[1].id == "z/doc"
