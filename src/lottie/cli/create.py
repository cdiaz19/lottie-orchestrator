"""`lottie create agent|skill <name>` — scaffold a complete unit from templates."""

from __future__ import annotations

from pathlib import Path

import typer

from lottie.scaffold.generator import generate

create_app = typer.Typer(help="Scaffold a new agent or skill.", no_args_is_help=True)


@create_app.command("agent")
def create_agent(name: str) -> None:
    """Scaffold a complete agent module."""
    target = generate("agent", name)
    rel = target.relative_to(Path.cwd())
    typer.echo(f"Created agent at {rel}")
    typer.echo(f"Next: implement {rel / 'agent.py'} then run `pytest {rel}`")


@create_app.command("skill")
def create_skill(name: str) -> None:
    """Scaffold a complete skill module."""
    target = generate("skill", name)
    rel = target.relative_to(Path.cwd())
    typer.echo(f"Created skill at {rel}")
    typer.echo(f"Next: implement {rel / 'skill.py'} then run `pytest {rel}`")
