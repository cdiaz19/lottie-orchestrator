"""index_manifest — warm a vector store from all documents in a KnowledgeManifest.

This helper is used by agents at startup to populate an in-process store from the
project's already-persisted knowledge files.  It is intentionally read-only over
the knowledge tree: it does NOT write drafts, run the ingest security gate, or
modify frontmatter.

Public API::

    from lottie.knowledge import index_manifest
"""

from __future__ import annotations

from lottie.knowledge.chunking import ChunkConfig, chunk_document
from lottie.knowledge.embeddings.base import EmbeddingProvider
from lottie.knowledge.manifest import KnowledgeManifest
from lottie.knowledge.schema import EmbeddedChunk
from lottie.knowledge.store.base import VectorStore


def index_manifest(
    manifest: KnowledgeManifest,
    embedder: EmbeddingProvider,
    store: VectorStore,
    *,
    config: ChunkConfig | None = None,
) -> int:
    """Chunk + embed every document in *manifest* and add it to *store*.

    Parameters
    ----------
    manifest:
        Loaded manifest produced by :meth:`KnowledgeManifest.from_root`.
        Only documents present in ``manifest.documents`` are indexed.
    embedder:
        Provider used to embed the chunk texts.  Must support arbitrary batch
        sizes (``embedder.embed([text, ...])``) and return one
        :class:`~lottie.knowledge.schema.Embedding` per input text.
    store:
        Target vector store.  All produced :class:`~lottie.knowledge.schema.EmbeddedChunk`
        objects are passed to ``store.add(...)`` in a single batch per document.
    config:
        Chunking configuration.  Defaults to ``ChunkConfig()`` (size=1000,
        overlap=200) when ``None``.

    Returns
    -------
    int
        Total number of chunks added across all documents.  Returns 0 when the
        manifest is empty or all documents have empty content.

    Notes
    -----
    - Read-only: no files are written; the ingest security gate is NOT invoked.
      The documents are assumed already vetted (they live in ``knowledge/``).
    - This is a warming helper intended for agent startup, not a replacement for
      the full ``DocumentIngestSkill`` ingest pipeline.
    """
    cfg = config or ChunkConfig()
    total = 0

    for doc in manifest.documents:
        chunks = chunk_document(doc, cfg)
        if not chunks:
            continue

        texts = [chunk.text for chunk in chunks]
        embeddings = embedder.embed(texts)

        items = [
            EmbeddedChunk(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        store.add(items)
        total += len(items)

    return total
