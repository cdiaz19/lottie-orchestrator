"""`lottie create agent|skill <name>` — scaffold a unit from templates or a description."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from lottie.llm import build_provider
from lottie.project.config import find_project_root, load_lottie_config
from lottie.scaffold.generator import Kind, generate, generate_from_desc

create_app = typer.Typer(help="Scaffold a new agent or skill.", no_args_is_help=True)

_FromDesc = Annotated[
    str | None,
    typer.Option("--from-desc", help="AI-generate the unit from this description."),
]


@create_app.command("agent")
def create_agent(name: str, from_desc: _FromDesc = None) -> None:
    """Scaffold an agent — from templates, or AI-generated with --from-desc."""
    _create("agent", name, from_desc)


@create_app.command("skill")
def create_skill(name: str, from_desc: _FromDesc = None) -> None:
    """Scaffold a skill — from templates, or AI-generated with --from-desc."""
    _create("skill", name, from_desc)


def _create(kind: Kind, name: str, from_desc: str | None) -> None:
    if from_desc is None:
        target = generate(kind, name)
        rel = target.relative_to(Path.cwd())
        typer.echo(f"Created {kind} at {rel}")
        typer.echo(f"Next: implement {rel / f'{kind}.py'} then run `pytest {rel}`")
        return
    root = find_project_root()
    llm = build_provider(load_lottie_config(root).providers.default)
    result = generate_from_desc(kind, name, from_desc, llm)
    location = f"{kind}s/{name}/"
    typer.echo(f"AI-generated {kind} at {location} ({len(result.files_written)} files)")
