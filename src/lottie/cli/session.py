"""`lottie session` — inspect and manage long-running session artifacts (V2 S5b)."""

from __future__ import annotations

import json

import typer

from lottie.project.config import find_project_root
from lottie.session.store import InvalidSessionId, SessionNotFound, SessionStore

session_app = typer.Typer(help="Inspect long-running agent sessions.", no_args_is_help=True)


@session_app.command("list")
def list_sessions() -> None:
    """List sessions with their agent, run count, and current progress keys."""
    store = SessionStore(find_project_root())
    ids = store.list()
    if not ids:
        typer.echo("no sessions")
        return
    for session_id in ids:
        state = store.require(session_id)
        keys = ", ".join(sorted(state.progress)) or "—"
        typer.echo(f"{session_id}  agent={state.agent}  runs={len(state.runs)}  progress=[{keys}]")


@session_app.command("show")
def show(session_id: str) -> None:
    """Print a session's full state as JSON."""
    store = SessionStore(find_project_root())
    try:
        state = store.require(session_id)
    except (SessionNotFound, InvalidSessionId) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(state.model_dump(), indent=2))


@session_app.command("delete")
def delete(session_id: str) -> None:
    """Remove a session and its artifacts."""
    store = SessionStore(find_project_root())
    try:
        removed = store.delete(session_id)
    except InvalidSessionId as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"deleted '{session_id}'" if removed else f"no session named '{session_id}'")
