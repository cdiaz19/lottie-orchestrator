"""Lottie-native REST transport over AgentService.

Pure wrapper, same shape as serve/openai_app.py. Imports Starlette at module top, so it is
imported lazily (never from serve/__init__.py) — the base install needs neither [serve] nor
[api]."""

from __future__ import annotations

import logging
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from lottie.serve.error_map import json_error
from lottie.serve.rest_schema import agent_detail_dict, agent_list_dict
from lottie.serve.service import AgentService

logger = logging.getLogger(__name__)


def rest_routes(svc: AgentService, root: Path) -> list[Route]:
    """The Lottie-native REST routes (/v1/agents[...]), closed over svc + root."""

    async def list_agents(request: Request) -> JSONResponse:
        return JSONResponse(agent_list_dict(svc.list_agents()))

    async def agent_detail(request: Request) -> JSONResponse:
        from lottie.project.config import load_agent_config
        from lottie.project.discovery import load_input_model

        name = request.path_params["name"]
        if not (root / "agents" / name / "agent.py").is_file():
            return json_error(404, f"agent '{name}' not found", type_="not_found")
        try:
            schema: dict[str, object] = load_input_model(root, name).model_json_schema()
            provider = load_agent_config(root / "agents" / name).provider
        except Exception:  # noqa: BLE001 — exists but won't introspect -> 500
            return json_error(500, "internal error", type_="internal_error")
        return JSONResponse(agent_detail_dict(name, provider, schema))

    return [
        Route("/v1/agents", list_agents, methods=["GET"]),
        Route("/v1/agents/{name}", agent_detail, methods=["GET"]),
    ]


def build_rest_app(root: Path, *, service: AgentService | None = None) -> Starlette:
    """Build a Starlette app exposing the REST routes (for isolated testing)."""
    svc = service or AgentService(root)
    return Starlette(routes=rest_routes(svc, root))
