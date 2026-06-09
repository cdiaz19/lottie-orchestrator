"""DocumentIngestSkill — load, gate, chunk, embed, and store knowledge documents.

CLAUDE.md rules enforced here:
- Rule 10: every source passes PromptInjectionScanSkill AND SecretDetectionSkill.
- Rule 12: documents are always written to ``knowledge/draft/``; promotion to
  ``curated`` requires a separate human step.

No LLM is used; the skill is fully deterministic for a fixed embedder + store.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from lottie.core import BaseSkill
from lottie.knowledge.chunking import chunk_document
from lottie.knowledge.embeddings import EmbeddingProvider
from lottie.knowledge.ingest import draft_filename, load_source, make_draft_id
from lottie.knowledge.schema import DocStatus, Document, EmbeddedChunk, KnowledgeLayer
from lottie.knowledge.store import VectorStore
from lottie.security import (
    InjectionScanInput,
    PromptInjectionScanSkill,
    ScanInput,
    SecretDetectionSkill,
)

from .schema import DocumentIngestInput, DocumentIngestOutput


class DocumentIngestSkill(BaseSkill[DocumentIngestInput, DocumentIngestOutput]):
    """Load file/text sources, run the injection+secret gate, chunk, embed, and store.

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

    # ------------------------------------------------------------------
    # Internal gate helpers
    # ------------------------------------------------------------------

    def _injection_flagged(self, text: str, source_label: str) -> bool:
        """Return True if *text* contains prompt-injection markers."""
        scanner = PromptInjectionScanSkill(enable_benchmarks=False)
        result = scanner.run(InjectionScanInput(content=text, source=source_label))
        return result.flagged

    def _secret_flagged(self, text: str) -> bool:
        """Return True if *text* contains detectable secrets.

        ``SecretDetectionSkill`` operates on file paths, so we write *text* to
        a named temporary file, scan it, and delete the file immediately.
        """
        detector = SecretDetectionSkill(enable_benchmarks=False)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            encoding="utf-8",
            delete=False,
        ) as tmp:
            tmp.write(text)
            tmp_path = tmp.name

        try:
            result = detector.run(ScanInput(paths=[tmp_path]))
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
        """Write a YAML-frontmatter markdown file under ``knowledge/draft/``."""
        draft_dir = self._root / "knowledge" / "draft"
        draft_dir.mkdir(parents=True, exist_ok=True)

        filename = draft_filename(draft_id)
        dest = draft_dir / filename

        frontmatter = (
            "---\n"
            f"id: {draft_id}\n"
            "layer: draft\n"
            "status: draft\n"
            f"target_layer: {target_layer.value}\n"
            "tags: []\n"
            "depends_on: []\n"
            "---\n"
        )
        dest.write_text(frontmatter + "\n" + text, encoding="utf-8")

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def _execute(self, data: DocumentIngestInput) -> DocumentIngestOutput:
        """Process each source through the security gate, then chunk + embed + store."""
        documents: list[Document] = []
        flagged: list[str] = []
        chunk_count = 0

        for source in data.sources:
            # 1. Load raw text.
            text = load_source(source)

            # 2. Derive a deterministic id for this source.
            draft_id = make_draft_id(source)

            # Choose a human-readable source label for the injection scanner.
            source_label = (
                source.value if source.kind == "file" else f"text:{draft_id}"
            )

            # 3. Security gate (CLAUDE.md rule 10 — both scans, no exceptions).
            if self._injection_flagged(text, source_label):
                flagged.append(draft_id)
                continue

            if self._secret_flagged(text):
                flagged.append(draft_id)
                continue

            # 4. Build Document (always DRAFT — rule 12).
            doc = Document(
                id=draft_id,
                source=source_label,
                layer=KnowledgeLayer.DRAFT,
                content=text,
                status=DocStatus.DRAFT,
                frontmatter={"target_layer": source.layer.value},
            )

            # 5. Write draft file (rule 12 — write only to knowledge/draft/).
            self._write_draft(draft_id, text, source.layer)

            # 6. Chunk → embed → store.
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

        return DocumentIngestOutput(
            documents=documents,
            chunk_count=chunk_count,
            flagged=flagged,
        )
