"""`lottie serve` — run the MCP stdio server for the current project."""

from __future__ import annotations

import typer

from lottie.project.config import find_project_root


def serve() -> None:
    """Serve the project's agents as MCP tools over stdio."""
    try:
        from lottie.serve.mcp_server import serve_stdio
    except ImportError as exc:
        raise typer.BadParameter(
            "lottie serve needs the MCP SDK. "
            "Install: pip install lottie-orchestrator[serve]"
        ) from exc
    root = find_project_root()
    serve_stdio(root)
