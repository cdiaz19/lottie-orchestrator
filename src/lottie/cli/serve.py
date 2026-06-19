"""`lottie serve` — MCP stdio by default, or an OpenAI-compatible HTTP API with --port."""

from __future__ import annotations

import typer

from lottie.project.config import find_project_root


def serve(
    port: int | None = typer.Option(
        None, "--port", "-p", help="Serve the OpenAI-compatible HTTP API on this port."
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind the HTTP API."),
) -> None:
    """Serve the project's agents.

    No --port: MCP tools over stdio (needs [serve]). With --port: an
    OpenAI-compatible /v1/chat/completions HTTP API (needs [api]).
    """
    root = find_project_root()

    if port is None:
        try:
            from lottie.serve.mcp_server import serve_stdio
        except ImportError as exc:
            raise typer.BadParameter(
                "lottie serve needs the MCP SDK. "
                "Install: pip install lottie-orchestrator[serve]"
            ) from exc
        serve_stdio(root)
        return

    try:
        import uvicorn

        from lottie.serve.http_app import build_http_app
    except ImportError as exc:
        raise typer.BadParameter(
            "lottie serve --port needs the HTTP API deps. "
            "Install: pip install lottie-orchestrator[api]"
        ) from exc
    uvicorn.run(build_http_app(root), host=host, port=port)
