"""`lottie mesh` — CLI sub-commands for mesh pause/resume and checkpoint history.

Commands
--------
resume   Resume an interrupted mesh agent thread with an approve/reject decision.
history  Report what checkpoint history is (and is not) available from the CLI.

Cross-process limitation
-------------------------
The default mesh checkpointer is an in-memory ``MemorySaver``, whose state is
**process-local**. A fresh ``lottie mesh resume`` process starts with an empty
checkpointer and therefore cannot see a thread that was interrupted by a *prior*
``lottie run`` process. In-memory resume only works *within the same process*
(e.g. the embedded MCP server). Durable cross-process resume requires the
sqlite checkpointer (the ``[mesh]`` extra) and is out of scope for these
commands — so this CLI focuses on wiring: command existence, argument
validation, and helpful errors rather than a true cross-process round-trip.

This module imports cleanly *without* the ``[mesh]`` extra: langgraph is only
reached lazily, through ``AgentService`` (which is langgraph-free; meshes import
langgraph behind a guard).
"""

from __future__ import annotations

import json
from typing import Annotated, Literal, cast

import typer
from rich.console import Console
from rich.panel import Panel

from lottie.mesh.errors import MeshError
from lottie.mesh.schema import ApprovalDecision
from lottie.project.config import find_project_root
from lottie.serve.schema import RunResult
from lottie.serve.service import AgentService, ServeError

mesh_app = typer.Typer(
    help=(
        "Resume interrupted mesh threads and inspect checkpoint history. "
        "NOTE: the default in-memory checkpointer is process-local, so "
        "cross-process resume is not possible — use the sqlite checkpointer "
        "for durable, cross-process state."
    ),
    no_args_is_help=True,
)

_console = Console()

_VALID_DECISIONS = {"approve", "reject"}

# Install/limitation hint reused across error paths.
_MESH_HINT = (
    "Install the mesh extra (pip install 'lottie-orchestrator[mesh]'). "
    "In-memory resume only works within the same process; use the sqlite "
    "checkpointer for durable cross-process state."
)


@mesh_app.command("resume")
def resume(
    agent: Annotated[
        str,
        typer.Option("--agent", help="Name of the mesh agent to resume."),
    ],
    thread_id: Annotated[
        str,
        typer.Option("--thread-id", help="Thread id of the interrupted run."),
    ],
    decision: Annotated[
        str,
        typer.Option("--decision", help="Human approval decision: 'approve' or 'reject'."),
    ] = "approve",
    input_json: Annotated[
        str,
        typer.Option(
            "--input",
            help="JSON object of edited input to apply on approve (default: none).",
        ),
    ] = "",
) -> None:
    """Resume an interrupted mesh agent thread with an approve/reject decision.

    NOTE: the default in-memory checkpointer is process-local. A fresh CLI
    process cannot see a thread interrupted by a prior ``lottie run`` process,
    so this only succeeds within the same process. For durable cross-process
    resume use the sqlite checkpointer.
    """
    if decision not in _VALID_DECISIONS:
        raise typer.BadParameter(
            f"decision must be 'approve' or 'reject', got {decision!r}.",
            param_hint="--decision",
        )

    edited_input: dict[str, str] = {}
    if input_json:
        try:
            parsed = json.loads(input_json)
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(
                f"--input must be valid JSON: {exc}", param_hint="--input"
            ) from exc
        if not isinstance(parsed, dict):
            raise typer.BadParameter(
                "--input must be a JSON object.", param_hint="--input"
            )
        edited_input = {str(k): str(v) for k, v in parsed.items()}

    root = find_project_root()
    action = cast("Literal['approve', 'reject']", decision)
    decision_obj = ApprovalDecision(action=action, edited_input=edited_input)

    try:
        svc = AgentService(root)
        result = svc.resume_agent(agent, thread_id, decision_obj)
    except ImportError as exc:
        _console.print(f"[red]Error:[/red] {exc}\n{_MESH_HINT}")
        raise typer.Exit(1) from exc
    except (ServeError, MeshError) as exc:
        _console.print(
            f"[red]Error:[/red] could not resume '{agent}' (thread {thread_id!r}): {exc}\n"
            f"{_MESH_HINT}"
        )
        raise typer.Exit(1) from exc

    _print_result(result)


@mesh_app.command("history")
def history(
    agent: Annotated[
        str,
        typer.Option("--agent", help="Name of the mesh agent."),
    ],
    thread_id: Annotated[
        str,
        typer.Option("--thread-id", help="Thread id to inspect."),
    ],
) -> None:
    """Report what checkpoint history is available from the CLI.

    Honest by design: with the default in-memory checkpointer, checkpoint
    history is process-local, so a cold CLI invocation has nothing to show for
    a thread created in another process. Validates the project root and prints
    an informative notice; exits 0.
    """
    # Validate that we are inside a Lottie project (mirrors other commands).
    find_project_root()

    body = (
        f"agent:      {agent}\n"
        f"thread_id:  {thread_id}\n\n"
        "Checkpoint history requires the [mesh] extra and a persistent (sqlite) "
        "checkpointer for cross-process inspection.\n"
        "The default in-memory checkpointer is process-local: a fresh CLI process "
        "starts with an empty MemorySaver, so there is nothing to show across "
        "processes for a thread created by a prior run.\n\n"
        "Switch to the sqlite checkpointer for durable, inspectable history."
    )
    _console.print(Panel(body, title="mesh history (in-memory: nothing across processes)"))


def _print_result(result: RunResult) -> None:
    """Render a RunResult from a resume as a rich panel."""
    body = (
        f"agent:       {result.agent}\n"
        f"status:      {result.status}\n"
        f"thread_id:   {result.thread_id or '—'}\n"
        f"latency_ms:  {result.latency_ms:.1f}\n"
        f"tokens:      in={result.input_tokens} out={result.output_tokens}\n"
        f"cost_usd:    {result.cost_usd:.6f}\n"
        f"pending:     {result.pending or '—'}\n\n"
        f"output:\n{json.dumps(result.output, indent=2, default=str)}"
    )
    _console.print(Panel(body, title=f"resume: {result.agent}"))
