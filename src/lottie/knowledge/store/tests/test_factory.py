"""Tests for the ``build_vector_store`` factory.

Covers:
- memory backend (always available, no extra required)
- chroma backend (guarded by importorskip, needs [chroma] extra)
- unknown kind raises ValueError
- lazy-fail path: simulate chromadb absence and assert the helpful ImportError

The lazy-fail test does NOT require chromadb to be installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from lottie.knowledge.store import InMemoryVectorStore, build_vector_store


class TestBuildVectorStoreMemory:
    """memory backend works without any optional extra."""

    def test_returns_in_memory_store(self, tmp_path: Path) -> None:
        store = build_vector_store("memory", tmp_path)
        assert isinstance(store, InMemoryVectorStore)


class TestBuildVectorStoreUnknown:
    """Unknown kind raises ValueError with a useful message."""

    def test_unknown_kind_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown vector store kind"):
            build_vector_store("bogus", tmp_path)


class TestBuildVectorStoreChromaPresent:
    """chroma backend: only runs when chromadb is installed."""

    chromadb = pytest.importorskip("chromadb")

    def test_returns_chroma_vector_store(self, tmp_path: Path) -> None:
        from lottie.knowledge.store import VectorStore
        from lottie.knowledge.store.chroma import ChromaVectorStore

        store = build_vector_store("chroma", tmp_path)
        assert isinstance(store, VectorStore)
        assert isinstance(store, ChromaVectorStore)


class TestBuildVectorStoreChromaMissing:
    """Lazy-fail: when chromadb is absent the factory raises a helpful ImportError.

    We simulate the absence by inserting ``None`` for the submodule in
    ``sys.modules`` — Python treats ``sys.modules[name] = None`` as a
    "module not found" sentinel and raises ``ImportError`` on import.
    This test does NOT need chromadb to be installed.
    """

    def test_chroma_missing_gives_helpful_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Evict any already-imported chroma module so the factory re-tries the import.
        monkeypatch.setitem(sys.modules, "lottie.knowledge.store.chroma", None)
        with pytest.raises(ImportError, match=r"lottie-orchestrator\[chroma\]"):
            build_vector_store("chroma", tmp_path)
