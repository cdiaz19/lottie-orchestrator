"""KnowledgeManifest — filesystem loader for the knowledge/ tree.

Walks ``<root>/knowledge/<layer>/`` for each :class:`~lottie.knowledge.schema.KnowledgeLayer`
and builds a list of :class:`~lottie.knowledge.schema.Document` objects using the YAML
frontmatter parser.  The directory is the authoritative source for the layer — the ``layer:``
field inside a file's frontmatter is ignored by this loader.

Example::

    manifest = KnowledgeManifest.from_root(Path("."))
    doc = manifest.by_id("lottie/auth-conventions")

Filesystem-only: reads YAML and plain text; does **not** import user code.
"""

from __future__ import annotations

from pathlib import Path

from lottie.knowledge.frontmatter import to_document
from lottie.knowledge.schema import Document, KnowledgeLayer


class KnowledgeManifest:
    """Loaded snapshot of all knowledge documents found under a project root.

    Attributes
    ----------
    documents:
        All documents discovered, sorted by :attr:`~lottie.knowledge.schema.Document.id`
        for deterministic ordering.
    """

    def __init__(self, documents: list[Document]) -> None:
        self.documents: list[Document] = documents

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_root(cls, root: Path) -> KnowledgeManifest:
        """Walk ``<root>/knowledge/<layer>/`` for every :class:`KnowledgeLayer`.

        Parameters
        ----------
        root:
            Project root (or fixtures root in tests).  The loader looks for a
            ``knowledge/`` subdirectory directly inside *root*; missing layer dirs
            are silently skipped.

        Returns
        -------
        KnowledgeManifest
            Instance whose :attr:`documents` are sorted by ``id`` for determinism.
        """
        knowledge_root = root / "knowledge"
        docs: list[Document] = []

        for layer in KnowledgeLayer:
            layer_dir = knowledge_root / layer.value
            if not layer_dir.is_dir():
                continue
            for path in sorted(layer_dir.rglob("*.md")):
                raw = path.read_text(encoding="utf-8")
                doc = to_document(path, layer, raw)
                docs.append(doc)

        docs.sort(key=lambda d: d.id)
        return cls(docs)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def by_id(self, id: str) -> Document | None:
        """Return the first document whose :attr:`~Document.id` equals *id*, or ``None``."""
        for doc in self.documents:
            if doc.id == id:
                return doc
        return None

    def by_layer(self, layer: KnowledgeLayer) -> list[Document]:
        """Return all documents whose :attr:`~Document.layer` equals *layer*."""
        return [doc for doc in self.documents if doc.layer == layer]
