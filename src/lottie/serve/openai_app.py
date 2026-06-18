"""OpenAI-compatible HTTP transport over AgentService.

Pure wrapper, same shape as serve/mcp_server.py. Imports Starlette at module top,
so it is imported lazily (never from serve/__init__.py) — the base install needs
neither [serve] nor [api].
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from lottie.project.config import ChatConfig, load_agent_config
from lottie.serve.service import AgentService

logger = logging.getLogger(__name__)


def _chat_config(root: Path, name: str) -> ChatConfig | None:
    """The agent's chat block, or None if the agent is missing/unloadable/not chat."""
    if not (root / "agents" / name / "agent.py").is_file():
        return None
    try:
        return load_agent_config(root / "agents" / name).chat
    except Exception as exc:  # noqa: BLE001 — a broken agent is simply not chat-exposed
        logger.warning("skipping agent %r for chat: %s", name, exc)
        return None


def build_openai_app(root: Path, *, service: AgentService | None = None) -> Starlette:
    """Build a Starlette app exposing chat-capable agents over the OpenAI API."""
    svc = service or AgentService(root)

    async def list_models(request: Request) -> JSONResponse:
        from lottie.project.discovery import discover_agents

        created = int(time.time())
        data = [
            {"id": unit.name, "object": "model", "created": created, "owned_by": "lottie"}
            for unit in discover_agents(root)
            if _chat_config(root, unit.name) is not None
        ]
        return JSONResponse({"object": "list", "data": data})

    app = Starlette(routes=[Route("/v1/models", list_models, methods=["GET"])])
    app.state.svc = svc  # POST route (next task) picks this up from request.app.state.svc
    return app
