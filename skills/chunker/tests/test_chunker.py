"""Unit tests for ChunkerSkill and chunk_document (deterministic — no LLM)."""

from __future__ import annotations

from lottie.knowledge.chunking import ChunkConfig, chunk_document
from lottie.knowledge.schema import Document, KnowledgeLayer
from skills.chunker.schema import ChunkerInput, ChunkerOutput
from skills.chunker.skill import ChunkerSkill

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(content: str, doc_id: str = "d") -> Document:
    return Document(id=doc_id, source="s", layer=KnowledgeLayer.GLOBAL, content=content)


# ---------------------------------------------------------------------------
# chunk_document — core algorithm tests
# ---------------------------------------------------------------------------


def test_determinism_and_offsets() -> None:
    """2500-char doc, default config → 3 chunks; second chunk starts at 800."""
    doc = _doc("a" * 2500)
    cfg = ChunkConfig()  # size=1000, overlap=200
    chunks = chunk_document(doc, cfg)

    assert len(chunks) == 3
    # second chunk starts at end_of_chunk0 - overlap
    # chunk0: start=0, end=1000 (no snap possible — "a" has no separator)
    # chunk1: start = max(1000-200, 0+1) = 800
    assert chunks[1].start == 800
    # ids
    assert [c.id for c in chunks] == ["d#0", "d#1", "d#2"]
    # determinism: two runs give identical (id, start, end)
    run1 = [(c.id, c.start, c.end) for c in chunks]
    run2 = [(c.id, c.start, c.end) for c in chunk_document(doc, cfg)]
    assert run1 == run2


def test_boundary_snapping() -> None:
    """Separator \\n\\n inside window causes end to snap past the separator."""
    content = ("x" * 500) + "\n\n" + ("y" * 2000)
    doc = _doc(content)
    cfg = ChunkConfig()  # size=1000, overlap=200

    chunks = chunk_document(doc, cfg)

    # The "\n\n" starts at index 500; just after it is index 502.
    # chunk_document should snap end to 502 (last "\n\n" inside [0, 1000]).
    assert chunks[0].end == 502


def test_empty_content() -> None:
    """Empty document → no chunks."""
    doc = _doc("")
    assert chunk_document(doc, ChunkConfig()) == []


def test_metadata_on_every_chunk() -> None:
    """Every chunk carries layer and doc_id in metadata."""
    doc = _doc("a" * 2500)
    chunks = chunk_document(doc, ChunkConfig())

    for chunk in chunks:
        assert chunk.metadata["layer"] == "global"
        assert chunk.metadata["doc_id"] == "d"


def test_single_chunk_short_doc() -> None:
    """Doc shorter than chunk size → exactly one chunk."""
    doc = _doc("hello world")
    chunks = chunk_document(doc, ChunkConfig(size=1000, overlap=200))

    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].start == 0
    assert chunks[0].end == len("hello world")


def test_chunk_text_matches_slice() -> None:
    """chunk.text must equal doc.content[start:end] for every chunk."""
    content = "a" * 2500
    doc = _doc(content)
    for chunk in chunk_document(doc, ChunkConfig()):
        assert chunk.text == content[chunk.start : chunk.end]


def test_custom_config_respected() -> None:
    """Custom size + overlap changes chunk count and start offsets."""
    doc = _doc("a" * 1000)
    cfg = ChunkConfig(size=400, overlap=100)
    chunks = chunk_document(doc, cfg)

    # chunk0: [0, 400); chunk1: [300, 700); chunk2: [600, 1000)
    assert chunks[0].start == 0
    assert chunks[0].end == 400
    assert chunks[1].start == 300
    assert chunks[2].start == 600


# ---------------------------------------------------------------------------
# ChunkerSkill — skill-layer tests
# ---------------------------------------------------------------------------


def test_skill_returns_chunker_output() -> None:
    """ChunkerSkill.run returns a ChunkerOutput with the expected chunks."""
    doc = _doc("a" * 2500)
    skill = ChunkerSkill()

    result = skill.run(ChunkerInput(document=doc))

    assert isinstance(result, ChunkerOutput)
    assert len(result.chunks) == 3


def test_skill_chunks_match_direct_call() -> None:
    """Skill result equals direct chunk_document call (same deterministic output)."""
    doc = _doc("a" * 2500)
    cfg = ChunkConfig(size=500, overlap=100)
    skill = ChunkerSkill()

    skill_chunks = skill.run(ChunkerInput(document=doc, config=cfg)).chunks
    direct_chunks = chunk_document(doc, cfg)

    assert [(c.id, c.start, c.end) for c in skill_chunks] == [
        (c.id, c.start, c.end) for c in direct_chunks
    ]


def test_skill_last_metrics_populated() -> None:
    """last_metrics is not None after run — skill is benchmarkable from day one."""
    doc = _doc("hello world")
    skill = ChunkerSkill()
    skill.run(ChunkerInput(document=doc))

    assert skill.last_metrics is not None


def test_skill_empty_doc() -> None:
    """Empty document → ChunkerOutput with empty chunks list."""
    doc = _doc("")
    skill = ChunkerSkill()

    result = skill.run(ChunkerInput(document=doc))

    assert result.chunks == []
