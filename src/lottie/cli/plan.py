"""`lottie plan` — inspect and replay a mesh run's recorded routing (E6).

A mesh routes dynamically: the supervisor decides each step from what already happened.
That makes a multi-agent flow non-deterministic and therefore hard to test or debug — you
cannot re-run "the same" flow. A recorded plan fixes that: the decisions a run actually
made, replayable with zero supervisor calls.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from lottie.mesh.plan import PlanDivergence, PlanNotFound, list_plans, load_plan
from lottie.project.config import find_project_root

plan_app = typer.Typer(
    help="Inspect and replay recorded mesh execution plans.", no_args_is_help=True
)


@plan_app.command("list")
def list_command(agent: str) -> None:
    """List runs of `agent` that have a recorded plan."""
    root = find_project_root()
    threads = list_plans(root, agent)
    if not threads:
        typer.echo(f"no recorded plans for '{agent}'")
        return
    for thread in threads:
        plan = load_plan(root, agent, thread)
        typer.echo(f"{thread}  {len(plan.steps)} step(s)")


@plan_app.command("show")
def show(agent: str, thread_id: str) -> None:
    """Render a recorded plan as the sequence of routing decisions it made."""
    root = find_project_root()
    try:
        plan = load_plan(root, agent, thread_id)
    except (PlanNotFound, PlanDivergence) as exc:
        raise typer.BadParameter(str(exc)) from exc

    table = Table(title=f"{agent} — recorded plan ({thread_id})")
    table.add_column("Step", justify="right")
    table.add_column("Workers")
    table.add_column("Mode")
    for step in plan.steps:
        parallel = len(step.workers) > 1
        table.add_row(
            str(step.step),
            ", ".join(step.workers),
            "[cyan]parallel[/cyan]" if parallel else "sequential",
        )
    Console().print(table)
    # The task is stored hash-only, so a plan is safe to share and to keep. Say so, since
    # an operator reasonably wonders where the task went.
    typer.echo(
        f"task sha256: {plan.task_sha256[:16]}…  "
        "(hash only — the task text is never stored)"
    )
