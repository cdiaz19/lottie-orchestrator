"""Tests for MockEmbeddingProvider — deterministic, no real API calls."""

from __future__ import annotations

import math

import pytest

from lottie.knowledge.embeddings.mock import MockEmbeddingProvider


def l2_norm(v: list[float]) -> float:
    total: float = math.fsum(x * x for x in v)
    return math.sqrt(total)


class TestMockEmbeddingProviderDefaults:
    """Tests using the default dim=16, model="mock/embed" constructor."""

    def setup_method(self) -> None:
        self.provider = MockEmbeddingProvider()

    def test_same_text_same_vector(self) -> None:
        """embed called with the same text twice must return identical vectors."""
        results = self.provider.embed(["a", "a", "b"])
        assert results[0].vector == results[1].vector

    def test_different_text_different_vector(self) -> None:
        """embed called with different texts must (almost surely) return different vectors."""
        results = self.provider.embed(["a", "a", "b"])
        assert results[0].vector != results[2].vector

    def test_vector_length_is_dim(self) -> None:
        """Each returned Embedding has len(vector) == dim."""
        results = self.provider.embed(["hello", "world"])
        for emb in results:
            assert len(emb.vector) == 16

    def test_embedding_dim_field(self) -> None:
        """Each returned Embedding has .dim == 16."""
        results = self.provider.embed(["hello"])
        assert results[0].dim == 16

    def test_embedding_model_field(self) -> None:
        """Each returned Embedding has .model == 'mock/embed'."""
        results = self.provider.embed(["hello"])
        assert results[0].model == "mock/embed"

    def test_vector_is_l2_normalized(self) -> None:
        """L2 norm of each vector must be approximately 1.0."""
        results = self.provider.embed(["alpha", "beta", "gamma"])
        for emb in results:
            norm = l2_norm(emb.vector)
            assert norm == pytest.approx(1.0, abs=1e-6)

    def test_model_property(self) -> None:
        """model property returns the configured model string."""
        assert self.provider.model == "mock/embed"


class TestMockEmbeddingProviderDeterminism:
    """Determinism tests across separate provider instances."""

    def test_deterministic_across_instances(self) -> None:
        """Two fresh providers must produce identical vectors for the same text."""
        v1 = MockEmbeddingProvider().embed(["x"])[0].vector
        v2 = MockEmbeddingProvider().embed(["x"])[0].vector
        assert v1 == v2

    def test_deterministic_different_texts_across_instances(self) -> None:
        """Determinism holds for a non-trivial text string."""
        text = "the quick brown fox"
        v1 = MockEmbeddingProvider().embed([text])[0].vector
        v2 = MockEmbeddingProvider().embed([text])[0].vector
        assert v1 == v2


class TestMockEmbeddingProviderCustomDim:
    """Tests with non-default dim values."""

    def test_custom_dim_8_field(self) -> None:
        """dim=8 returns Embedding with .dim == 8."""
        provider = MockEmbeddingProvider(dim=8)
        result = provider.embed(["q"])[0]
        assert result.dim == 8

    def test_custom_dim_8_vector_length(self) -> None:
        """dim=8 returns a vector of length 8."""
        provider = MockEmbeddingProvider(dim=8)
        result = provider.embed(["q"])[0]
        assert len(result.vector) == 8

    def test_custom_dim_8_normalized(self) -> None:
        """dim=8 vector is L2-normalized."""
        provider = MockEmbeddingProvider(dim=8)
        result = provider.embed(["q"])[0]
        assert l2_norm(result.vector) == pytest.approx(1.0, abs=1e-6)

    def test_custom_model_string(self) -> None:
        """Custom model string is reflected in Embedding.model."""
        provider = MockEmbeddingProvider(model="test/embed-v2")
        result = provider.embed(["z"])[0]
        assert result.model == "test/embed-v2"

    def test_dim_64_vector_length(self) -> None:
        """dim=64 (>32 bytes from one SHA-256 digest) returns length-64 vector."""
        provider = MockEmbeddingProvider(dim=64)
        result = provider.embed(["extended"])[0]
        assert len(result.vector) == 64

    def test_dim_64_normalized(self) -> None:
        """dim=64 vector is L2-normalized."""
        provider = MockEmbeddingProvider(dim=64)
        result = provider.embed(["extended"])[0]
        assert l2_norm(result.vector) == pytest.approx(1.0, abs=1e-6)

    def test_dim_64_deterministic(self) -> None:
        """dim=64 is deterministic across instances."""
        v1 = MockEmbeddingProvider(dim=64).embed(["extended"])[0].vector
        v2 = MockEmbeddingProvider(dim=64).embed(["extended"])[0].vector
        assert v1 == v2
