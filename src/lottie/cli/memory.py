"""`lottie memory` — CLI sub-commands for knowledge dependency-graph queries.

Commands
--------
graph   Print the dependency graph edges and a node/edge count summary.
impact  Show which documents transitively depend on a given document id.
audit   Report cycles (hard failure), orphans, and stale documents.

Note: "memory" here refers to the *knowledge dependency graph* (spec §15
``lottie memory graph/impact/audit``).  It is entirely distinct from the
runtime agent memory subsystem in ``lottie.memory``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from lottie.knowledge import GraphStore, KnowledgeManifest

memory_app = typer.Typer(
    help=(
        "Query the knowledge dependency graph "
        "(distinct from runtime agent memory)."
    ),
    no_args_is_help=True,
)

_console = Console()

# ---------------------------------------------------------------------------
# Shared option type alias
# ---------------------------------------------------------------------------

_RootOption = Annotated[
    Path,
    typer.Option("--root", help="Project root containing knowledge/."),
]


# ---------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------


@memory_app.command("graph")
def graph(
    root: _RootOption = Path("."),
) -> None:
    """Print the knowledge dependency graph edges and a summary.

    Edges are oriented as ``dependency -> dependent`` (if B depends on A,
    the edge is A -> B).  The summary reports total node and edge counts.
    """
    manifest = KnowledgeManifest.from_root(root)
    store = GraphStore(manifest)
    g = store.graph

    node_count = g.number_of_nodes()
    edge_count = g.number_of_edges()

    if node_count == 0:
        _console.print("Knowledge graph is empty.")
        return

    table = Table(title="Knowledge Dependency Graph")
    table.add_column("dependency", style="cyan")
    table.add_column("→", justify="center")
    table.add_column("dependent", style="green")

    for dep, dependent in sorted(g.edges()):
        table.add_row(dep, "→", dependent)

    _console.print(table)
    _console.print(
        f"[bold]{node_count}[/bold] node(s), [bold]{edge_count}[/bold] edge(s)."
    )


# ---------------------------------------------------------------------------
# impact
# ---------------------------------------------------------------------------


@memory_app.command("impact")
def impact(
    doc_id: Annotated[
        str,
        typer.Argument(help="Document id to query (e.g. global/auth-conventions)."),
    ],
    root: _RootOption = Path("."),
) -> None:
    """Show which documents (transitively) depend on DOC_ID.

    Answers "what breaks if DOC_ID is deprecated?"  Exits 0 in all cases,
    including when DOC_ID is not present in the graph (informational query).
    """
    manifest = KnowledgeManifest.from_root(root)
    store = GraphStore(manifest)

    if doc_id not in store.graph:
        _console.print(f"No document '{doc_id}' in the knowledge graph.")
        return

    dependents = store.impact(doc_id)

    if not dependents:
        _console.print(f"Nothing depends on '{doc_id}'.")
        return

    table = Table(title=f"Dependents of '{doc_id}'")
    table.add_column("dependent id", style="green")
    for dep_id in dependents:
        table.add_row(dep_id)

    _console.print(table)
    _console.print(
        f"[bold]{len(dependents)}[/bold] document(s) transitively depend on '{doc_id}'."
    )


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


@memory_app.command("audit")
def audit(
    root: _RootOption = Path("."),
    stale_days: Annotated[
        int,
        typer.Option(
            "--stale-days",
            help="Age threshold in days for stale documents (default: 90).",
        ),
    ] = 90,
) -> None:
    """Audit the knowledge graph for cycles, orphans, and stale documents.

    Exit codes
    ----------
    * 0 — no cycles detected (orphans and stale are warnings only).
    * 1 — one or more cycles found (hard failure, CI can gate on this).
    """
    manifest = KnowledgeManifest.from_root(root)
    store = GraphStore(manifest)

    found_cycles = store.cycles()
    found_orphans = store.orphans()
    found_stale = store.stale(stale_days)

    # --- Cycles ---
    if found_cycles:
        _console.print(
            f"[bold red]✗ {len(found_cycles)} cycle(s) found:[/bold red]"
        )
        for cycle in found_cycles:
            cycle_str = " → ".join(cycle) + f" → {cycle[0]}"
            _console.print(f"  [red]•[/red] {cycle_str}")
    else:
        _console.print("[bold green]✓ no cycles[/bold green]")

    # --- Orphans ---
    if found_orphans:
        _console.print(
            f"\n[yellow]⚠ {len(found_orphans)} orphan(s) "
            f"(no edges — not linked to or from anything):[/yellow]"
        )
        for oid in found_orphans:
            _console.print(f"  • {oid}")
    else:
        _console.print("\n[dim]No orphans.[/dim]")

    # --- Stale ---
    if found_stale:
        _console.print(
            f"\n[yellow]⚠ {len(found_stale)} stale document(s) "
            f"(last_verified > {stale_days} days ago):[/yellow]"
        )
        for sid in found_stale:
            _console.print(f"  • {sid}")
    else:
        _console.print(f"\n[dim]No stale documents (threshold: {stale_days} days).[/dim]")

    if found_cycles:
        raise typer.Exit(1)
