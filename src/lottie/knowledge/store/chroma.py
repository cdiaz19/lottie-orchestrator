"""Persistent ChromaDB vector store backend.

Uses ``chromadb.PersistentClient`` to store embeddings under
``root / ".lottie" / "chroma"``.  The collection is created with cosine
distance (``hnsw:space = cosine``) so that distances are in ``[0, 2]`` and
we convert to a similarity score via ``score = 1.0 - distance``.

We own all embeddings — Chroma never computes them.  Every ``add`` call
passes ``embeddings=`` explicitly and uses the default ``EmbeddingFunction``
(None) on the collection.

Lossless round-trip
-------------------
Chroma metadata values must be ``str | int | float | bool``.  To avoid
lossy serialisation of the full ``Chunk`` model, we stash
``chunk.model_dump_json()`` as a ``_chunk_json`` metadata key.  On
``query``, we reconstruct each ``Chunk`` via
``Chunk.model_validate_json(metadata["_chunk_json"])``.

Tag filtering
-------------
Tags are stored as a CSV string (``metadata["tags"]``).  Chroma cannot
filter on CSV substrings server-side, so tag filtering is applied
client-side *after* fetching results from Chroma.  When a tag filter is
active, we fetch all candidates (``count()`` total) before filtering to
minimise the chance of under-delivering.  When no tag filter is active,
we pass ``n_results=k`` directly — Chroma caps results at collection size
automatically, so no extra ``count()`` RPC is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
import chromadb.api

from lottie.knowledge.schema import (
    Chunk,
    EmbeddedChunk,
    Embedding,
    KnowledgeLayer,
    RetrievalHit,
)
from lottie.knowledge.store.base import VectorStore

__all__ = ["ChromaVectorStore"]


def _parse_tags(raw: str) -> set[str]:
    """Split a CSV tag string and return a set of stripped, non-empty tags."""
    return {t.strip() for t in raw.split(",") if t.strip()}


class ChromaVectorStore(VectorStore):
    """Persistent vector store backed by ChromaDB.

    Parameters
    ----------
    root:
        Project root directory.  Chroma data is stored under
        ``root / ".lottie" / "chroma"``.
    collection:
        Chroma collection name.  Defaults to ``"lottie_knowledge"``.
    """

    def __init__(
        self,
        root: Path,
        *,
        collection: str = "lottie_knowledge",
    ) -> None:
        self._collection_name = collection
        persist_dir = root / ".lottie" / "chroma"
        persist_dir.mkdir(parents=True, exist_ok=True)
        # PersistentClient is a factory function that returns ClientAPI.
        self._client: chromadb.api.ClientAPI = chromadb.PersistentClient(
            path=str(persist_dir)
        )
        self._collection: Any = self._open_collection()


    def _open_collection(self) -> Any:
        """Get or create the collection and assert it uses cosine distance.

        ``get_or_create_collection`` silently ignores the ``metadata`` arg
        when the collection already exists, so a pre-existing collection with
        a different ``hnsw:space`` (e.g. ``"l2"``) would produce silently wrong
        similarity scores.  We detect and reject that case early.
        """
        coll: Any = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        actual: str = (coll.metadata or {}).get("hnsw:space", "l2")
        if actual != "cosine":
            raise RuntimeError(
                f"Chroma collection {self._collection_name!r} uses "
                f"hnsw:space={actual!r}, expected 'cosine'. "
                "Delete .lottie/chroma to reset."
            )
        return coll


    def add(self, items: list[EmbeddedChunk]) -> None:
        """Persist *items* to the Chroma collection.

        Empty batches are a no-op (Chroma raises on empty ``add``).

        Note on duplicate IDs: re-adding a chunk with an existing id is
        backend-defined — Chroma keeps the first write (ignores duplicates)
        while ``InMemoryVectorStore`` appends; callers must ``clear()`` and
        re-add to update an existing chunk.
        """
        if not items:
            return

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, str]] = []

        for item in items:
            if "_chunk_json" in item.chunk.metadata:
                raise ValueError(
                    "chunk.metadata may not use the reserved key '_chunk_json'"
                )
            ids.append(item.chunk.id)
            embeddings.append(list(item.embedding.vector))
            documents.append(item.chunk.text)
            # Stash the full serialised Chunk so query can reconstruct losslessly.
            metadatas.append(
                {
                    **item.chunk.metadata,
                    "_chunk_json": item.chunk.model_dump_json(),
                }
            )

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        embedding: Embedding,
        k: int,
        *,
        layers: list[KnowledgeLayer] | None = None,
        tags: list[str] | None = None,
    ) -> list[RetrievalHit]:
        """Return the top-*k* chunks most similar to *embedding*.

        Layer filtering is applied server-side via a Chroma ``where`` clause.
        Tag filtering is applied client-side after fetching candidates.

        See ``VectorStore.query`` for the full contract.
        """
        if k <= 0:
            return []

        # Build the server-side where clause for layer filtering.
        effective_layers: list[str] = (
            [layer.value for layer in layers] if layers else []
        )
        where: dict[str, object] | None = (
            {"layer": {"$in": effective_layers}} if effective_layers else None
        )

        # Normalise the tag filter: drop empty/whitespace strings.
        tag_set: set[str] | None = None
        if tags is not None:
            stripped = {t.strip() for t in tags if t.strip()}
            tag_set = stripped if stripped else None

        # Decide how many candidates to fetch.
        # When a tag filter is active we overfetch all documents (client-side
        # filtering may otherwise under-deliver), so we need the count.
        # When no tag filter is active, pass k directly — Chroma caps
        # n_results at the collection size automatically, saving an RPC.
        if tag_set is not None:
            total: int = self._collection.count()
            if total == 0:
                return []
            fetch_n: int = total  # overfetch all, then client-side filter + truncate to k
        else:
            fetch_n = k  # Chroma caps n_results at collection size automatically

        query_kwargs: dict[str, object] = {
            "query_embeddings": [list(embedding.vector)],
            "n_results": fetch_n,
            "include": ["metadatas", "distances"],
        }
        if where is not None:
            query_kwargs["where"] = where

        results: Any = self._collection.query(**query_kwargs)

        # results["metadatas"] is [[{...}, ...]], results["distances"] is [[float, ...]]
        raw_metadatas: list[Any] = results["metadatas"][0]
        raw_distances: list[float] = results["distances"][0]

        hits: list[RetrievalHit] = []
        for metadata, distance in zip(raw_metadatas, raw_distances, strict=True):
            meta: dict[str, str] = dict(metadata)  # normalise from Chroma mapping type
            # Client-side tag filter.
            if tag_set is not None:
                chunk_tags = _parse_tags(meta.get("tags", ""))
                if not chunk_tags.intersection(tag_set):
                    continue

            # Reconstruct the full Chunk losslessly from the stashed JSON.
            chunk = Chunk.model_validate_json(meta["_chunk_json"])

            # Convert Chroma cosine distance to similarity: score = 1 - distance.
            score: float = 1.0 - float(distance)
            hits.append(RetrievalHit(chunk=chunk, score=score))

        # Chroma already returns nearest-first; preserve that order after tag
        # filtering, then truncate to k.
        return hits[:k]

    def count(self) -> int:
        """Return the total number of chunks in the collection."""
        return int(self._collection.count())

    def clear(self) -> None:
        """Remove all chunks from the collection.

        Deletes the collection and immediately recreates it (empty) so that
        subsequent calls to ``add`` and ``query`` work normally.
        """
        self._client.delete_collection(self._collection_name)
        self._collection = self._open_collection()
