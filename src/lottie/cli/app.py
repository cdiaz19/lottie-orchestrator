"""Lottie CLI — Typer application.

Single `app` instance exposed as the `lottie` console script. Subcommands
live in sibling modules and register here.
"""

from __future__ import annotations

import typer

from lottie.cli.audit import audit
from lottie.cli.benchmark import benchmark_app
from lottie.cli.create import create_app
from lottie.cli.distill import distill
from lottie.cli.doctor import doctor
from lottie.cli.init import init
from lottie.cli.knowledge import knowledge_app
from lottie.cli.memory import memory_app
from lottie.cli.mesh import mesh_app
from lottie.cli.reflect import reflect
from lottie.cli.registry import inspect_app, list_app
from lottie.cli.run import run
from lottie.cli.serve import serve
from lottie.cli.status import status

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
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(create_app, name="create")
app.add_typer(knowledge_app, name="knowledge")
app.add_typer(list_app, name="list")
app.add_typer(memory_app, name="memory")
app.add_typer(mesh_app, name="mesh")
app.add_typer(inspect_app, name="inspect")
app.command("distill")(distill)
app.command("reflect")(reflect)
app.command("run")(run)
app.command("serve")(serve)
app.command("status")(status)
app.command("audit")(audit)
app.command("doctor")(doctor)
