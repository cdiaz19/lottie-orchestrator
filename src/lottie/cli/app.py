"""Lottie CLI — Typer application.

Single `app` instance exposed as the `lottie` console script. Subcommands
live in sibling modules and register here.
"""

from __future__ import annotations

import typer

from lottie.cli.create import create_app
from lottie.cli.init import init

app = typer.Typer(
    help="Lottie AI Orchestrator",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _main() -> None:
    # No-op root callback. Forces Typer into multi-command (group) mode so
    # `lottie <command>` parses correctly even with a single registered command.
    pass


app.command("init")(init)
app.add_typer(create_app, name="create")
