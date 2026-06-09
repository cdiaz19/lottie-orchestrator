"""`lottie knowledge` — CLI sub-commands for the knowledge layer.

Commands
--------
ingest   Load text/file/URL sources into the draft knowledge layer.
list     Show all knowledge documents as a rich table.
inspect  Show metadata and chunk count for a single document.
clear    Delete draft (or other-layer) files from the filesystem.

Provider defaults
-----------------
- ``--embedder`` defaults to ``LOTTIE_EMBEDDING_MODEL`` env var or
  ``"mock/embed"`` (deterministic, no API key required).  For production,
  pass ``--embedder openai/text-embedding-3-small``.
- ``--store`` defaults to ``LOTTIE_VECTOR_STORE`` env var or ``"memory"``.
  Use ``"chroma"`` for persistent storage (requires ``[chroma]`` extra).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lottie.knowledge import GraphStore, KnowledgeManifest
from lottie.knowledge.chunking import ChunkConfig, chunk_document
from lottie.knowledge.embeddings import build_embedding_provider
from lottie.knowledge.ingest import DocumentIngestInput, DocumentIngestSkill, IngestSource
from lottie.knowledge.schema import KnowledgeLayer
from lottie.knowledge.store import build_vector_store

knowledge_app = typer.Typer(
    help="Manage the knowledge layer: ingest, list, inspect, and clear documents.",
    no_args_is_help=True,
)

_console = Console()

# ---------------------------------------------------------------------------
# Shared option defaults
# ---------------------------------------------------------------------------

# Sentinel: empty string means "resolve from env at runtime" so that
# monkeypatch.setenv works in tests and env vars are read at call time,
# not at module import time.
_EMBEDDER_SENTINEL = ""
_STORE_SENTINEL = ""


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


@knowledge_app.command("ingest")
def ingest(
    root: Annotated[
        Path,
        typer.Option("--root", help="Project root containing knowledge/."),
    ] = Path("."),
    embedder: Annotated[
        str,
        typer.Option(
            "--embedder",
            help=(
                "Embedding model identifier.  Defaults to LOTTIE_EMBEDDING_MODEL "
                "env var or 'mock/embed' (no API key needed).  Production: "
                "'openai/text-embedding-3-small'."
            ),
        ),
    ] = _EMBEDDER_SENTINEL,
    store: Annotated[
        str,
        typer.Option(
            "--store",
            help=(
                "Vector store backend ('memory' or 'chroma').  Defaults to "
                "LOTTIE_VECTOR_STORE env var or 'memory'."
            ),
        ),
    ] = _STORE_SENTINEL,
    layer: Annotated[
        str,
        typer.Option("--layer", help="Target knowledge layer (default: draft)."),
    ] = "draft",
    file: Annotated[
        list[Path] | None,
        typer.Option("--file", help="File path(s) to ingest (repeatable)."),
    ] = None,
    text: Annotated[
        str | None,
        typer.Option("--text", help="Raw text to ingest."),
    ] = None,
    url: Annotated[
        str | None,
        typer.Option("--url", help="URL to ingest (deferred — not yet implemented)."),
    ] = None,
) -> None:
    """Load file/text/URL sources into the draft knowledge layer.

    At least one of --file, --text, or --url must be provided.
    """
    # Validate at least one source.
    if not file and text is None and url is None:
        _console.print("[red]Error:[/red] provide at least one of --file, --text, or --url.")
        raise typer.Exit(1)

    # Resolve env-var defaults at call time (not import time) so that
    # monkeypatch.setenv works correctly in tests.
    effective_embedder = embedder or os.environ.get("LOTTIE_EMBEDDING_MODEL", "mock/embed")
    effective_store = store or os.environ.get("LOTTIE_VECTOR_STORE", "memory")

    # Validate layer before touching the filesystem or building providers.
    try:
        target_layer = KnowledgeLayer(layer)
    except ValueError as exc:
        _console.print(f"[red]Error:[/red] unknown layer {layer!r}.")
        raise typer.Exit(1) from exc

    # Build provider objects — wrap in try/except for friendly error messages.
    try:
        embedder_provider = build_embedding_provider(effective_embedder)
        vector_store = build_vector_store(effective_store, root)
    except (ValueError, ImportError) as exc:
        _console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    sources: list[IngestSource] = []
    for f in file or []:
        sources.append(IngestSource(kind="file", value=str(f), layer=target_layer))
    if text is not None:
        sources.append(IngestSource(kind="text", value=text, layer=target_layer))
    if url is not None:
        sources.append(IngestSource(kind="url", value=url, layer=target_layer))

    # Run the skill.
    skill = DocumentIngestSkill(embedder_provider, vector_store, root)
    result = skill.run(DocumentIngestInput(sources=sources, config=ChunkConfig()))

    # Report.
    _console.print(
        f"[green]Ingested:[/green] {len(result.documents)} document(s), "
        f"{result.chunk_count} chunk(s)."
    )
    if result.flagged:
        _console.print(
            f"[yellow]Flagged (security gate):[/yellow] {len(result.flagged)} source(s)"
        )
        for fid in result.flagged:
            _console.print(f"  • {fid}")
    if result.errors:
        _console.print(f"[red]Errors:[/red] {len(result.errors)} source(s) failed to load.")
        for err in result.errors:
            _console.print(f"  • {err}")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@knowledge_app.command("list")
def list_docs(
    root: Annotated[
        Path,
        typer.Option("--root", help="Project root containing knowledge/."),
    ] = Path("."),
) -> None:
    """Show all knowledge documents as a rich table."""
    manifest = KnowledgeManifest.from_root(root)

    if not manifest.documents:
        _console.print("No knowledge documents found.")
        return

    table = Table(title="Knowledge Documents")
    table.add_column("id")
    table.add_column("layer")
    table.add_column("status")
    table.add_column("#tags")
    table.add_column("#depends_on")

    for doc in manifest.documents:
        table.add_row(
            doc.id,
            doc.layer.value,
            doc.status.value,
            str(len(doc.tags)),
            str(len(doc.depends_on)),
        )

    _console.print(table)


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


@knowledge_app.command("inspect")
def inspect_doc(
    doc_id: Annotated[str, typer.Argument(help="Document id to inspect (e.g. draft/text_abc123).")],
    root: Annotated[
        Path,
        typer.Option("--root", help="Project root containing knowledge/."),
    ] = Path("."),
) -> None:
    """Show metadata and chunk count for a single document."""
    manifest = KnowledgeManifest.from_root(root)
    doc = manifest.by_id(doc_id)

    if doc is None:
        typer.echo(f"Error: document '{doc_id}' not found.", err=True)
        raise typer.Exit(1)

    chunks = chunk_document(doc, ChunkConfig())
    graph = GraphStore(manifest)
    dependents = graph.impact(doc_id)

    body = (
        f"id:          {doc.id}\n"
        f"layer:       {doc.layer.value}\n"
        f"status:      {doc.status.value}\n"
        f"source:      {doc.source}\n"
        f"tags:        {doc.tags or '—'}\n"
        f"depends_on:  {doc.depends_on or '—'}\n"
        f"chunks:      {len(chunks)}\n"
        f"dependents:  {dependents or '—'}"
    )
    _console.print(Panel(body, title=f"document: {doc_id}"))


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


@knowledge_app.command("clear")
def clear_docs(
    root: Annotated[
        Path,
        typer.Option("--root", help="Project root containing knowledge/."),
    ] = Path("."),
    layer: Annotated[
        str,
        typer.Option("--layer", help="Layer to clear (default: draft)."),
    ] = "draft",
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt."),
    ] = False,
) -> None:
    """Delete *.md files under <root>/knowledge/<layer>/.

    Targets the draft layer by default (CLAUDE.md rule 12 — agents write only
    to draft; promotion to curated requires human review).  The directory
    itself is NOT removed.
    """
    # Validate layer against the enum BEFORE touching the filesystem.
    # This rejects path traversal attempts like --layer "../../etc".
    try:
        KnowledgeLayer(layer)
    except ValueError as exc:
        _console.print(
            f"[red]Error:[/red] {layer!r} is not a valid knowledge layer. "
            f"Valid layers: {', '.join(v.value for v in KnowledgeLayer)}."
        )
        raise typer.Exit(1) from exc

    layer_dir = root / "knowledge" / layer

    # Defense-in-depth: ensure the resolved path stays inside knowledge/.
    knowledge_root = root.resolve() / "knowledge"
    if not layer_dir.resolve().is_relative_to(knowledge_root):
        _console.print(
            "[red]Error:[/red] resolved layer path escapes the knowledge directory."
        )
        raise typer.Exit(1)

    if not layer_dir.is_dir():
        _console.print(f"Removed 0 file(s) from '{layer}'.")
        return

    md_files = list(layer_dir.glob("*.md"))
    count = len(md_files)

    if count == 0:
        _console.print(f"Removed 0 file(s) from '{layer}'.")
        return

    if not yes:
        confirmed = typer.confirm(
            f"Delete {count} file(s) from '{layer_dir}'?", default=False
        )
        if not confirmed:
            _console.print("Aborted.")
            raise typer.Exit(0)

    for f in md_files:
        f.unlink()

    _console.print(f"Removed {count} file(s) from '{layer}'.")
