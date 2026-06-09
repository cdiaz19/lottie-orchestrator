"""litellm-backed `EmbeddingProvider`.

litellm is Lottie's universal proxy: one interface to OpenAI, Cohere, Voyage,
local models, etc.  This adapter is the ONLY place litellm is imported for
embeddings — agent and skill code always goes through the `EmbeddingProvider`
abstraction (CLAUDE.md rule 1), mirroring how `lottie.llm.litellm_provider`
handles completions.
"""

from __future__ import annotations

import litellm
from litellm.types.utils import EmbeddingResponse

from lottie.knowledge.embeddings.base import EmbeddingProvider
from lottie.knowledge.schema import Embedding

_DEFAULT_MODEL = "openai/text-embedding-3-small"


class LiteLLMEmbeddingProvider(EmbeddingProvider):
    """Routes embedding calls through litellm to any supported backend."""

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def embed(self, texts: list[str]) -> list[Embedding]:
        """Embed a batch of texts and return one `Embedding` per input string.

        If *texts* is empty the method returns ``[]`` immediately without
        calling litellm — avoids a pointless (possibly chargeable) API round-
        trip and keeps parity with ``MockEmbeddingProvider``.

        litellm normalises embedding responses to an OpenAI-style object whose
        ``.data`` attribute is a list of items each carrying an ``"embedding"``
        key (``list[float]``) and an optional ``"index"`` key (``int``).
        Items are sorted by ``"index"`` before mapping so that out-of-order
        delivery from the upstream model does not silently scramble results.

        Exceptions raised by litellm propagate unchanged — callers decide how
        to handle provider errors.  This mirrors how ``LiteLLMProvider`` lets
        ``litellm.completion`` exceptions propagate in ``lottie.llm``.
        """
        if not texts:
            return []

        # litellm.embedding returns Union[EmbeddingResponse, Coroutine[...]] per
        # its stubs; synchronous usage always yields EmbeddingResponse.  We
        # access .data via the concrete type so mypy resolves the attribute —
        # mirroring how LiteLLMProvider accesses response fields directly after
        # litellm.completion().
        response: EmbeddingResponse = litellm.embedding(
            model=self._model, input=texts
        )

        # response.data is typed as List (unparameterised) in litellm stubs;
        # mypy accepts it as list[object] implicitly.
        data: list[object] = list(response.data)

        # Guard against silent text↔vector misalignment: litellm must return
        # exactly one embedding per input string.
        if len(data) != len(texts):
            raise ValueError(
                f"litellm returned {len(data)} embeddings for {len(texts)} inputs"
            )

        # Sort by "index" to restore input order when items arrive shuffled.
        # The enumerate position is passed as the fallback so that items with
        # unparseable index values preserve their original relative order.
        indexed: list[tuple[int, object]] = list(enumerate(data))
        indexed.sort(key=lambda t: _item_index(t[1], t[0]))

        embeddings: list[Embedding] = []
        for _, item in indexed:
            vector = _extract_vector(item)
            embeddings.append(Embedding(vector=vector, model=self._model, dim=len(vector)))
        return embeddings


def _item_index(item: object, position: int) -> int:
    """Extract the ``'index'`` field from a litellm embedding data item.

    litellm guarantees items arrive in input order but carries an explicit
    ``"index"`` field for safety.  We extract it tolerantly so both dict-like
    and attribute-style objects work.

    *position* is the item's enumerate position in ``response.data``; it is
    returned as a fallback whenever the ``"index"`` field is absent or cannot
    be parsed as an integer.  This ensures ``sort`` is total and never raises.
    """
    try:
        return int(item["index"])  # type: ignore[index]  # dict-like access on opaque object
    except (TypeError, KeyError, ValueError):
        pass
    try:
        return int(getattr(item, "index", position))
    except (TypeError, ValueError):
        return position


def _extract_vector(item: object) -> list[float]:
    """Pull the float vector from a litellm embedding data item.

    litellm may return Pydantic-model objects, ``SimpleNamespace``-like
    objects, or plain dicts depending on version and backend.  We try two
    access patterns in order of reliability.
    """
    # 1. dict-style subscript (most common: plain dicts / dict-like objects)
    try:
        return list(item["embedding"])  # type: ignore[index]  # dict-like access on opaque object
    except (TypeError, KeyError):
        pass

    # 2. Attribute access (Pydantic model or SimpleNamespace)
    vector = getattr(item, "embedding", None)
    if vector is not None:
        return list(vector)

    raise ValueError(f"litellm embedding item missing 'embedding' field: {item!r}")
