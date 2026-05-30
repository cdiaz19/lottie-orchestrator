"""`lottie status` — show project config, registered units, and knowledge size."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from lottie.project.config import find_project_root, load_lottie_config
from lottie.project.discovery import UnitInfo, discover_agents, discover_skills


def status() -> None:
    """Show registered agents, skills, knowledge size, and provider config."""
    root = find_project_root()
    cfg = load_lottie_config(root)
    console = Console()
    console.print(f"[bold]{cfg.project}[/bold]")
    console.print(
        f"providers: default={cfg.providers.default} "
        f"fallback={cfg.providers.fallback or '—'}"
    )
    console.print(f"policies: {', '.join(cfg.policies) or '—'}")
    _print_units(console, "Agents", discover_agents(root))
    _print_units(console, "Skills", discover_skills(root))
    _print_knowledge(console, root)


def _print_units(console: Console, title: str, units: list[UnitInfo]) -> None:
    if not units:
        console.print(f"\n[bold]{title}[/bold]: _No {title.lower()} yet._")
        return
    table = Table(title=title)
    table.add_column("name")
    table.add_column("provider")
    for unit in units:
        table.add_row(unit.name, unit.provider or "—")
    console.print(table)


def _print_knowledge(console: Console, root: Path) -> None:
    kdir = root / "knowledge"
    if not kdir.is_dir():
        return
    console.print("\n[bold]Knowledge[/bold]")
    for layer in sorted(p for p in kdir.iterdir() if p.is_dir()):
        count = sum(1 for f in layer.iterdir() if f.is_file() and f.name != ".gitkeep")
        console.print(f"  {layer.name}: {count}")
