"""Lottie-native REST transport over AgentService.

Pure wrapper, same shape as serve/openai_app.py. Imports Starlette at module top, so it is
imported lazily (never from serve/__init__.py) — the base install needs neither [serve] nor
[api]."""

from __future__ import annotations

import logging
from pathlib import Path

import anyio
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from lottie.serve.error_map import json_error
from lottie.serve.errors import (
    InputSecurityViolation,
    NotResumable,
    OutputSecurityViolation,
    ThreadNotFound,
)
from lottie.serve.rest_schema import (
    ResumeRequest,
    agent_detail_dict,
    agent_list_dict,
    run_result_dict,
    withheld_dict,
)
from lottie.serve.service import (
    AgentExecutionError,
    AgentLoadError,
    AgentNotFoundError,
    AgentService,
    InvalidInputError,
)

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

    async def run_agent_route(request: Request) -> JSONResponse:
        name = request.path_params["name"]
        try:
            body = await request.json()
        except ValueError:
            return json_error(400, "invalid request body", type_="invalid_request")
        if not isinstance(body, dict):
            return json_error(400, "request body must be a JSON object", type_="invalid_request")

        try:
            result = await anyio.to_thread.run_sync(lambda: svc.run_agent(name, body))
        except InputSecurityViolation:
            return json_error(
                400, "request blocked by content policy", type_="content_filter"
            )
        except OutputSecurityViolation as exc:
            return JSONResponse(
                withheld_dict(name, input_tokens=exc.input_tokens, output_tokens=exc.output_tokens)
            )
        except InvalidInputError:
            return json_error(400, f"input does not fit agent '{name}'", type_="invalid_request")
        except AgentNotFoundError:
            return json_error(404, f"agent '{name}' not found", type_="not_found")
        except (AgentLoadError, AgentExecutionError):
            return json_error(500, "internal error", type_="internal_error")

        return JSONResponse(run_result_dict(result))

    async def resume_agent_route(request: Request) -> JSONResponse:
        name = request.path_params["name"]
        try:
            body = await request.json()
            req = ResumeRequest.model_validate(body)
        except (ValueError, ValidationError):
            return json_error(400, "invalid request body", type_="invalid_request")
        try:
            result = await anyio.to_thread.run_sync(
                lambda: svc.resume_agent(name, req.thread_id, req.decision)
            )
        except InputSecurityViolation:
            return json_error(400, "request blocked by content policy", type_="content_filter")
        except OutputSecurityViolation as exc:
            return JSONResponse(
                withheld_dict(name, input_tokens=exc.input_tokens, output_tokens=exc.output_tokens)
            )
        except NotResumable:
            return json_error(400, f"agent '{name}' is not resumable", type_="not_resumable")
        except ThreadNotFound:
            return json_error(404, "thread not found", type_="thread_not_found")
        except AgentNotFoundError:
            return json_error(404, f"agent '{name}' not found", type_="not_found")
        except (AgentLoadError, AgentExecutionError):
            return json_error(500, "internal error", type_="internal_error")
        return JSONResponse(run_result_dict(result))

    return [
        Route("/v1/agents", list_agents, methods=["GET"]),
        Route("/v1/agents/{name}", agent_detail, methods=["GET"]),
        Route("/v1/agents/{name}/run", run_agent_route, methods=["POST"]),
        Route("/v1/agents/{name}/resume", resume_agent_route, methods=["POST"]),
    ]


def build_rest_app(root: Path, *, service: AgentService | None = None) -> Starlette:
    """Build a Starlette app exposing the REST routes (for isolated testing)."""
    svc = service or AgentService(root)
    return Starlette(routes=rest_routes(svc, root))
