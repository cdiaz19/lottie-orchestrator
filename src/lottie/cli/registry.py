"""`lottie list` and `lottie inspect` — registry query commands."""

from __future__ import annotations

import typer
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lottie.project.config import find_project_root, load_agent_config
from lottie.project.discovery import (
    discover_agents,
    discover_skills,
    load_schema_models,
    load_system_prompt,
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


def _field_lines(model: type[BaseModel]) -> str:
    """One `name: type` line per field, for inspect output."""
    lines: list[str] = []
    for fname, field in model.model_fields.items():
        ann = field.annotation
        type_name: str = getattr(ann, "__name__", None) or str(ann)
        lines.append(f"  {fname}: {type_name}")
    return "\n".join(lines) or "  (no fields)"


@inspect_app.command("agent")
def inspect_agent(name: str) -> None:
    """Show an agent's config, schema, and system prompt."""
    root = find_project_root()
    if name not in [u.name for u in discover_agents(root)]:
        raise typer.BadParameter(f"agent '{name}' not found")
    cfg = load_agent_config(root / "agents" / name)
    in_model, out_model = load_schema_models(root, "agent", name)
    prompt = load_system_prompt(root, name) or "—"
    body = (
        f"provider: {cfg.provider}\n"
        f"model_params: {cfg.model_params}\n"
        f"capabilities: {', '.join(cfg.capabilities) or '—'}\n"
        f"policies: {', '.join(cfg.policies) or '—'}\n\n"
        f"Input:\n{_field_lines(in_model)}\n"
        f"Output:\n{_field_lines(out_model)}\n\n"
        f"System prompt:\n{prompt}"
    )
    Console().print(Panel(body, title=f"agent: {name}"))


@inspect_app.command("skill")
def inspect_skill(name: str) -> None:
    """Show a skill's schema and SKILL.md presence."""
    root = find_project_root()
    if name not in [u.name for u in discover_skills(root)]:
        raise typer.BadParameter(f"skill '{name}' not found")
    in_model, out_model = load_schema_models(root, "skill", name)
    has_doc = (root / "skills" / name / "SKILL.md").is_file()
    body = (
        f"SKILL.md: {'present' if has_doc else 'missing'}\n\n"
        f"Input:\n{_field_lines(in_model)}\n"
        f"Output:\n{_field_lines(out_model)}"
    )
    Console().print(Panel(body, title=f"skill: {name}"))
