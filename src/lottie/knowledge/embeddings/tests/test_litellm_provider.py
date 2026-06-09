"""Tests for LiteLLMEmbeddingProvider and the build_embedding_provider factory.

All tests are purely unit-level — no network, no real litellm calls.
``litellm.embedding`` is monkeypatched throughout.

Mirrors the testing style of ``tests/test_mock.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from lottie.knowledge.embeddings import (
    EmbeddingProvider,
    MockEmbeddingProvider,
    build_embedding_provider,
)
from lottie.knowledge.embeddings.litellm_provider import LiteLLMEmbeddingProvider

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "openai/text-embedding-3-small"
_STUB_VECTORS = [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
]


class _FakeEmbeddingResponse:
    """Mimics the shape of a litellm embedding response (OpenAI-style)."""

    def __init__(self, data: list[dict[str, Any]]) -> None:
        self.data = data


def _make_fake_embed(
    *,
    expected_model: str = _DEFAULT_MODEL,
    call_counter: list[int] | None = None,
    shuffle: bool = False,
) -> Any:
    """Return a fake ``litellm.embedding`` callable.

    Parameters
    ----------
    expected_model:
        The model string the fake asserts it receives.
    call_counter:
        If provided, the fake increments ``call_counter[0]`` on each call.
    shuffle:
        If True, the fake returns data items in *reverse* order (index still
        present) so the adapter's sort-by-index logic is exercised.
    """

    def fake_embedding(model: str, input: list[str], **kwargs: Any) -> _FakeEmbeddingResponse:  # noqa: A002
        if call_counter is not None:
            call_counter[0] += 1
        assert model == expected_model, f"Expected model {expected_model!r}, got {model!r}"
        data = [
            {"embedding": list(_STUB_VECTORS[i % len(_STUB_VECTORS)]), "index": i}
            for i in range(len(input))
        ]
        if shuffle:
            data = list(reversed(data))
        return _FakeEmbeddingResponse(data)

    return fake_embedding


# ---------------------------------------------------------------------------
# LiteLLMEmbeddingProvider — basic behaviour
# ---------------------------------------------------------------------------


class TestLiteLLMEmbeddingProviderModelProperty:
    """model property reflects the configured id."""

    def test_default_model(self) -> None:
        provider = LiteLLMEmbeddingProvider()
        assert provider.model == _DEFAULT_MODEL

    def test_custom_model(self) -> None:
        provider = LiteLLMEmbeddingProvider(model="cohere/embed-english-v3.0")
        assert provider.model == "cohere/embed-english-v3.0"

    def test_model_property_matches_constructor(self) -> None:
        model_id = "voyage/voyage-2"
        provider = LiteLLMEmbeddingProvider(model=model_id)
        assert provider.model == model_id


class TestLiteLLMEmbeddingProviderEmbed:
    """Core embed() behaviour — no network."""

    def test_embed_two_texts_returns_two_embeddings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """embed(['a', 'b']) returns exactly 2 Embedding objects."""
        monkeypatch.setattr("litellm.embedding", _make_fake_embed())
        provider = LiteLLMEmbeddingProvider()
        results = provider.embed(["a", "b"])
        assert len(results) == 2

    def test_embed_vectors_match_stub(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returned vectors match what the fake litellm response provides."""
        monkeypatch.setattr("litellm.embedding", _make_fake_embed())
        provider = LiteLLMEmbeddingProvider()
        results = provider.embed(["a", "b"])
        assert results[0].vector == pytest.approx(_STUB_VECTORS[0])
        assert results[1].vector == pytest.approx(_STUB_VECTORS[1])

    def test_embed_dim_equals_vector_length(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Embedding.dim == len(Embedding.vector) for every result."""
        monkeypatch.setattr("litellm.embedding", _make_fake_embed())
        provider = LiteLLMEmbeddingProvider()
        results = provider.embed(["x", "y"])
        for emb in results:
            assert emb.dim == len(emb.vector)

    def test_embed_model_field_matches_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Embedding.model == the provider's model string."""
        monkeypatch.setattr("litellm.embedding", _make_fake_embed())
        provider = LiteLLMEmbeddingProvider()
        results = provider.embed(["hello"])
        assert results[0].model == _DEFAULT_MODEL

    def test_embed_custom_model_propagated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Adapter passes the configured model id to litellm."""
        custom = "cohere/embed-english-v3.0"
        monkeypatch.setattr(
            "litellm.embedding", _make_fake_embed(expected_model=custom)
        )
        provider = LiteLLMEmbeddingProvider(model=custom)
        results = provider.embed(["test"])
        assert results[0].model == custom

    def test_embed_order_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Input order is preserved in the output list."""
        monkeypatch.setattr("litellm.embedding", _make_fake_embed())
        provider = LiteLLMEmbeddingProvider()
        results = provider.embed(["a", "b", "c"])
        # Index 0 → STUB_VECTORS[0], index 1 → [1], index 2 → [2]
        assert results[0].vector == pytest.approx(_STUB_VECTORS[0])
        assert results[1].vector == pytest.approx(_STUB_VECTORS[1])
        assert results[2].vector == pytest.approx(_STUB_VECTORS[2])


class TestLiteLLMEmbeddingProviderEmptyInput:
    """embed([]) must return [] without calling litellm."""

    def test_empty_returns_empty_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        counter: list[int] = [0]
        monkeypatch.setattr(
            "litellm.embedding", _make_fake_embed(call_counter=counter)
        )
        provider = LiteLLMEmbeddingProvider()
        results = provider.embed([])
        assert results == []

    def test_empty_does_not_call_litellm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        counter: list[int] = [0]
        monkeypatch.setattr(
            "litellm.embedding", _make_fake_embed(call_counter=counter)
        )
        provider = LiteLLMEmbeddingProvider()
        provider.embed([])
        assert counter[0] == 0, "litellm.embedding must NOT be called for empty input"


class TestLiteLLMEmbeddingProviderIndexSort:
    """When litellm returns items out-of-order, adapter re-orders by index."""

    def test_shuffled_response_reorders_by_index(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fake returns data in reverse order; adapter must return original order."""
        monkeypatch.setattr("litellm.embedding", _make_fake_embed(shuffle=True))
        provider = LiteLLMEmbeddingProvider()
        results = provider.embed(["a", "b", "c"])
        # After re-ordering: index 0 → STUB_VECTORS[0], etc.
        assert results[0].vector == pytest.approx(_STUB_VECTORS[0])
        assert results[1].vector == pytest.approx(_STUB_VECTORS[1])
        assert results[2].vector == pytest.approx(_STUB_VECTORS[2])


class TestLiteLLMEmbeddingProviderErrorPropagation:
    """Exceptions from litellm propagate unchanged."""

    def test_litellm_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(model: str, input: list[str], **kwargs: object) -> object:  # noqa: A002
            raise RuntimeError("litellm unavailable")

        monkeypatch.setattr("litellm.embedding", boom)
        provider = LiteLLMEmbeddingProvider()
        with pytest.raises(RuntimeError, match="litellm unavailable"):
            provider.embed(["hello"])


# ---------------------------------------------------------------------------
# build_embedding_provider factory
# ---------------------------------------------------------------------------


class TestBuildEmbeddingProviderFactory:
    """build_embedding_provider dispatches correctly on model prefix."""

    def test_mock_prefix_returns_mock_provider(self) -> None:
        provider = build_embedding_provider("mock/embed")
        assert isinstance(provider, MockEmbeddingProvider)

    def test_bare_mock_returns_mock_provider(self) -> None:
        provider = build_embedding_provider("mock")
        assert isinstance(provider, MockEmbeddingProvider)

    def test_mock_slash_custom_returns_mock_provider(self) -> None:
        provider = build_embedding_provider("mock/my-custom-embed")
        assert isinstance(provider, MockEmbeddingProvider)

    def test_openai_model_returns_litellm_provider(self) -> None:
        provider = build_embedding_provider("openai/text-embedding-3-small")
        assert isinstance(provider, LiteLLMEmbeddingProvider)

    def test_cohere_model_returns_litellm_provider(self) -> None:
        provider = build_embedding_provider("cohere/embed-english-v3.0")
        assert isinstance(provider, LiteLLMEmbeddingProvider)

    def test_litellm_provider_model_string_set(self) -> None:
        model = "openai/text-embedding-3-small"
        provider = build_embedding_provider(model)
        assert isinstance(provider, LiteLLMEmbeddingProvider)
        assert provider.model == model

    def test_mock_provider_model_string_set(self) -> None:
        provider = build_embedding_provider("mock/embed")
        assert isinstance(provider, MockEmbeddingProvider)
        assert provider.model == "mock/embed"

    def test_factory_returns_embedding_provider_abc(self) -> None:
        """Both paths satisfy the EmbeddingProvider interface."""
        for model in ("mock/embed", "openai/text-embedding-3-small"):
            provider = build_embedding_provider(model)
            assert isinstance(provider, EmbeddingProvider)
