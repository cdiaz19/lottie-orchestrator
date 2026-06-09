"""Framework utilities for the document ingest pipeline.

Provides:
- ``IngestSource`` — Pydantic model describing a single content source.
- ``load_source`` — load raw text from a typed ``IngestSource``.
- ``make_draft_id`` — deterministic, filesystem-safe draft identifier derived
  from the source kind/value and content hash.  Used as the document ``id``
  and as the stem of the draft ``.md`` file written under ``knowledge/draft/``.
- ``DocumentIngestInput`` / ``DocumentIngestOutput`` — Pydantic I/O models for
  ``DocumentIngestSkill``.
- ``DocumentIngestSkill`` — load, gate, chunk, embed, and store knowledge
  documents.  Lives here (framework) so the CLI never imports from ``skills/``.

No LLM, no network (URL ingest is explicitly deferred).  ``load_source`` and
``make_draft_id`` are pure / deterministic: identical inputs always produce
identical outputs.
"""

from __future__ import annotations

import hashlib
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from lottie.knowledge.schema import KnowledgeLayer

# ---------------------------------------------------------------------------
# IngestSource model (lives in the framework — skills import from here)
# ---------------------------------------------------------------------------


class IngestSource(BaseModel):
    """A single source of content to ingest into the knowledge layer.

    Parameters
    ----------
    kind:
        ``"text"`` — raw string content; ``"file"`` — path on disk;
        ``"url"`` — remote URL (deferred, raises ``NotImplementedError``).
    value:
        The source payload: the raw text, a file path, or a URL.
    layer:
        Requested *eventual* layer after human promotion. Phase 1 always
        writes to ``KnowledgeLayer.DRAFT`` regardless of this value.
    """

    kind: Literal["file", "text", "url"]
    value: str
    layer: KnowledgeLayer = KnowledgeLayer.DRAFT


# ---------------------------------------------------------------------------
# Source loader
# ---------------------------------------------------------------------------


def load_source(source: IngestSource) -> str:
    """Return the raw text for *source*.

    Dispatch rules
    --------------
    - ``kind == "text"`` → return ``source.value`` as-is.
    - ``kind == "file"`` → read ``Path(source.value)`` with UTF-8 encoding.
    - ``kind == "url"``  → raise ``NotImplementedError`` (deferred to a later phase).

    Raises
    ------
    NotImplementedError
        When ``source.kind == "url"``.
    FileNotFoundError
        When ``source.kind == "file"`` and the path does not exist.
    """
    if source.kind == "text":
        return source.value
    if source.kind == "file":
        return Path(source.value).read_text(encoding="utf-8")
    # url — deferred
    raise NotImplementedError(
        f"URL ingest is deferred to a later phase (got: {source.value!r})"
    )


# ---------------------------------------------------------------------------
# Draft-id / filename helpers
# ---------------------------------------------------------------------------

_UNSAFE = re.compile(r"[^\w\-]")  # keep word chars and hyphens


def _slugify(text: str, max_len: int = 48) -> str:
    """Convert *text* to a lowercase filesystem-safe slug."""
    slug = _UNSAFE.sub("_", text.lower()).strip("_")
    # Collapse runs of underscores
    slug = re.sub(r"_+", "_", slug)
    return slug[:max_len]


def make_draft_id(source: IngestSource, content: str) -> str:
    """Return a deterministic, filesystem-safe draft id for *source*.

    Rules
    -----
    - ``kind == "file"`` → ``"draft/<slugified-stem>-<sha1(content)[:8]>"``
      using ``Path.stem`` for the slug and the first 8 hex chars of the SHA-1
      of the file *content*.  This prevents collisions when two files share the
      same stem but carry different content.
    - ``kind == "text"`` → ``"draft/text_<sha1(content)[:12]>"`` where the
      SHA-1 is computed over the UTF-8-encoded content (same as before).
    - ``kind == "url"``  → ``"draft/url_<sha1(content)[:12]>"`` (consistent
      but URLs are rejected by ``load_source`` before this is needed in
      practice).

    The function is **idempotent**: re-ingesting identical content from the
    same source always produces the same id.

    The returned id is always prefixed with ``"draft/"`` to make the target
    layer explicit and to satisfy CLAUDE.md rule 12 (agents write only to
    ``knowledge/draft/``).
    """
    sha1 = hashlib.sha1(content.encode("utf-8"), usedforsecurity=False).hexdigest()
    if source.kind == "file":
        stem = _slugify(Path(source.value).stem) or "file"
        return f"draft/{stem}-{sha1[:8]}"
    # text or url: hash the content
    prefix = "text" if source.kind == "text" else "url"
    return f"draft/{prefix}_{sha1[:12]}"


def draft_filename(draft_id: str) -> str:
    """Return the ``.md`` filename for a *draft_id* (strips the ``draft/`` prefix)."""
    stem = draft_id.removeprefix("draft/")
    return f"{stem}.md"


def _today_iso() -> str:
    """Return today's UTC date as an ISO 8601 string."""
    return datetime.now(UTC).date().isoformat()


# ---------------------------------------------------------------------------
# DocumentIngestInput / DocumentIngestOutput (I/O models)
# ---------------------------------------------------------------------------

# Import here to avoid a circular dependency: chunking → schema (fine),
# but schema never imports ingest.
from lottie.knowledge.chunking import ChunkConfig, chunk_document  # noqa: E402
from lottie.knowledge.schema import Document  # noqa: E402, F401


class DocumentIngestInput(BaseModel):
    """Input for DocumentIngestSkill."""

    sources: list[IngestSource]
    config: ChunkConfig = ChunkConfig()


class DocumentIngestOutput(BaseModel):
    """Output from DocumentIngestSkill."""

    documents: list[Document] = []
    chunk_count: int = 0
    flagged: list[str] = []
    """Draft IDs (with ``draft/`` prefix) of sources rejected by the security gate."""
    errors: list[str] = []
    """Load/processing failures (bad path, URL not implemented, empty content, etc.).
    Each entry is a string of the form ``"<source identifier>: <reason>"``.
    These are distinct from ``flagged`` (security rejections) — no security gate was
    reached for errored sources.
    """


# ---------------------------------------------------------------------------
# DocumentIngestSkill
# ---------------------------------------------------------------------------

# These imports are placed after the model definitions above to keep the
# module-level import order logical (models first, skill second) and to make
# it easy to see which framework sub-packages DocumentIngestSkill depends on.
from lottie.core import BaseSkill  # noqa: E402
from lottie.knowledge.embeddings import EmbeddingProvider  # noqa: E402
from lottie.knowledge.schema import DocStatus, EmbeddedChunk  # noqa: E402
from lottie.knowledge.store import VectorStore  # noqa: E402
from lottie.security import (  # noqa: E402
    InjectionScanInput,
    PromptInjectionScanSkill,
    ScanInput,
    SecretDetectionSkill,
)


class DocumentIngestSkill(BaseSkill[DocumentIngestInput, DocumentIngestOutput]):
    """Load file/text sources, run the injection+secret gate, chunk, embed, and store.

    CLAUDE.md rules enforced here:
    - Rule 10: every source passes PromptInjectionScanSkill AND SecretDetectionSkill.
    - Rule 12: documents are always written to ``knowledge/draft/``; promotion to
      ``curated`` requires a separate human step.

    No LLM is used; the skill is fully deterministic for a fixed embedder + store.

    Parameters
    ----------
    embedder:
        Provider that converts chunk text into dense ``Embedding`` vectors.
    store:
        Vector store that persists the ``EmbeddedChunk`` objects.
    root:
        Project root directory.  Draft files are written under
        ``root / "knowledge" / "draft" / <filename>.md``.
    name:
        Optional display name forwarded to ``InstrumentedRunnable``.
    enable_benchmarks:
        Overrides the ``LOTTIE_DISABLE_BENCHMARKS`` env-var check when set.
    benchmarks_root:
        Directory under which benchmark JSONL files are appended.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStore,
        root: Path,
        *,
        name: str | None = None,
        enable_benchmarks: bool | None = None,
        benchmarks_root: Path | None = None,
    ) -> None:
        super().__init__(
            name=name,
            enable_benchmarks=enable_benchmarks,
            benchmarks_root=benchmarks_root,
        )
        self._embedder = embedder
        self._store = store
        self._root = root
        # Instantiate scanners once (M1) — not per-source.
        self._injection_scanner = PromptInjectionScanSkill(enable_benchmarks=False)
        self._secret_detector = SecretDetectionSkill(enable_benchmarks=False)

    # ------------------------------------------------------------------
    # Internal gate helpers
    # ------------------------------------------------------------------

    def _injection_flagged(self, text: str, source_label: str) -> bool:
        """Return True if *text* contains prompt-injection markers."""
        result = self._injection_scanner.run(
            InjectionScanInput(content=text, source=source_label)
        )
        return result.flagged

    def _secret_flagged(self, text: str) -> bool:
        """Return True if *text* contains detectable secrets.

        ``SecretDetectionSkill`` operates on file paths, so we write *text* to
        a named temporary file, scan it, and delete the file immediately.
        The temp-file path is captured as the first statement inside the
        ``with`` block so the ``finally`` branch always has a defined path.
        """
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            encoding="utf-8",
            delete=False,
        ) as tmp:
            tmp_path = tmp.name  # captured before write (I2)
            tmp.write(text)

        try:
            result = self._secret_detector.run(ScanInput(paths=[tmp_path]))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return len(result.findings) > 0

    # ------------------------------------------------------------------
    # Draft file writer
    # ------------------------------------------------------------------

    def _write_draft(
        self,
        draft_id: str,
        text: str,
        target_layer: KnowledgeLayer,
    ) -> None:
        """Write a YAML-frontmatter markdown file under ``knowledge/draft/``.

        Rule 14 fields included: id, layer, scope, status, target_layer,
        tags, depends_on, last_verified.
        """
        draft_dir = self._root / "knowledge" / "draft"
        draft_dir.mkdir(parents=True, exist_ok=True)

        filename = draft_filename(draft_id)
        dest = draft_dir / filename

        frontmatter = (
            "---\n"
            f"id: {draft_id}\n"
            "layer: draft\n"
            "scope: draft\n"                          # I1 — added
            "status: draft\n"
            f"target_layer: {target_layer.value}\n"
            "tags: []\n"
            "depends_on: []\n"
            f"last_verified: {_today_iso()}\n"        # I1 — added
            "---\n"
        )
        dest.write_text(frontmatter + "\n" + text, encoding="utf-8")

    # ------------------------------------------------------------------
    # Source identifier helper
    # ------------------------------------------------------------------

    @staticmethod
    def _source_identifier(source: IngestSource, draft_id: str) -> str:
        """Return a human-readable label for error / injection messages."""
        return source.value if source.kind == "file" else f"text:{draft_id}"

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def _execute(self, data: DocumentIngestInput) -> DocumentIngestOutput:
        """Process each source through the security gate, then chunk + embed + store."""
        documents: list[Document] = []
        flagged: list[str] = []
        errors: list[str] = []
        chunk_count = 0

        for source in data.sources:
            # Derive a preliminary source label for error messages (before draft_id
            # is known).
            raw_label = source.value if source.kind == "file" else f"<{source.kind}>"

            try:
                # 1. Load raw text.
                text = load_source(source)

                # 2. Skip empty / whitespace-only content (M3).
                if not text.strip():
                    errors.append(f"{raw_label}: empty content")
                    continue

                # 3. Derive a deterministic id for this source (M2 — content-hash).
                draft_id = make_draft_id(source, text)

                # Choose a human-readable source label for the injection scanner.
                source_label = self._source_identifier(source, draft_id)

                # 4. Security gate (CLAUDE.md rule 10 — both scans, no exceptions).
                if self._injection_flagged(text, source_label):
                    flagged.append(draft_id)
                    continue

                if self._secret_flagged(text):
                    flagged.append(draft_id)
                    continue

                # 5. Build Document (always DRAFT — rule 12).
                doc = Document(
                    id=draft_id,
                    source=source_label,
                    layer=KnowledgeLayer.DRAFT,
                    content=text,
                    status=DocStatus.DRAFT,
                    frontmatter={"target_layer": source.layer.value},
                )

                # 6. Write draft file (rule 12 — write only to knowledge/draft/).
                self._write_draft(draft_id, text, source.layer)

                # 7. Chunk → embed → store.
                chunks = chunk_document(doc, data.config)
                if chunks:
                    texts = [c.text for c in chunks]
                    embeddings = self._embedder.embed(texts)
                    embedded = [
                        EmbeddedChunk(chunk=chunk, embedding=emb)
                        for chunk, emb in zip(chunks, embeddings, strict=True)
                    ]
                    self._store.add(embedded)
                    chunk_count += len(chunks)

                documents.append(doc)

            except Exception as exc:  # I3 — per-source error isolation
                errors.append(f"{raw_label}: {exc}")
                continue

        return DocumentIngestOutput(
            documents=documents,
            chunk_count=chunk_count,
            flagged=flagged,
            errors=errors,
        )
