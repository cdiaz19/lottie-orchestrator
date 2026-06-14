"""`lottie audit` — query the immutable agent-run audit log."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from lottie.governance.audit import SqliteAuditLogger
from lottie.project.config import find_project_root

_console = Console()


def audit(
    agent: Annotated[
        str | None, typer.Option("--agent", help="Filter to one agent name.")
    ] = None,
    since: Annotated[
        str | None, typer.Option("--since", help="ISO-8601 lower bound on timestamp.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Max rows.")] = 50,
) -> None:
    """Show recent agent-run audit records from .lottie/audit.db."""
    root = find_project_root()
    records = SqliteAuditLogger(root).query(agent=agent, since=since, limit=limit)
    if not records:
        _console.print("[dim]no audit records yet[/dim]")
        return
    table = Table(title="audit log")
    for col in ("ts", "agent", "status", "root", "tokens", "cost", "ms", "in/out"):
        table.add_column(col)
    for r in records:
        table.add_row(
            r.ts, r.agent, r.status, "Y" if r.root else "-",
            f"{r.input_tokens}/{r.output_tokens}", f"{r.cost_usd:.6f}",
            f"{r.latency_ms:.1f}", f"{r.input_sha256[:8]}/{(r.output_sha256 or '-')[:8]}",
        )
    _console.print(table)
