"""`lottie list` and `lottie inspect` — registry query commands."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from lottie.project.config import find_project_root
from lottie.project.discovery import (
    discover_agents,
    discover_skills,
    load_schema_models,
)

list_app = typer.Typer(help="List registered agents or skills.", no_args_is_help=True)
inspect_app = typer.Typer(help="Inspect an agent or skill.", no_args_is_help=True)


@list_app.command("agents")
def list_agents() -> None:
    """List registered agents with their provider."""
    root = find_project_root()
    units = discover_agents(root)
    console = Console()
    if not units:
        console.print("_No agents yet._")
        return
    table = Table(title="Agents")
    table.add_column("name")
    table.add_column("provider")
    for unit in units:
        table.add_row(unit.name, unit.provider or "—")
    console.print(table)


@list_app.command("skills")
def list_skills() -> None:
    """List registered skills with their input/output types."""
    root = find_project_root()
    units = discover_skills(root)
    console = Console()
    if not units:
        console.print("_No skills yet._")
        return
    table = Table(title="Skills")
    table.add_column("name")
    table.add_column("input")
    table.add_column("output")
    for unit in units:
        try:
            in_model, out_model = load_schema_models(root, "skill", unit.name)
            in_name, out_name = in_model.__name__, out_model.__name__
        except Exception:  # noqa: BLE001 — one broken skill must not crash the list
            in_name, out_name = "—", "—"
        table.add_row(unit.name, in_name, out_name)
    console.print(table)
