"""OpenAI-compatible HTTP transport over AgentService.

Pure wrapper, same shape as serve/mcp_server.py. Imports Starlette at module top,
so it is imported lazily (never from serve/__init__.py) — the base install needs
neither [serve] nor [api].
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import anyio
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from lottie.project.config import ChatConfig, load_agent_config
from lottie.serve.error_map import json_error
from lottie.serve.errors import InputSecurityViolation, OutputSecurityViolation
from lottie.serve.openai_schema import (
    ChatCompletionRequest,
    chat_completion_dict,
    last_user_message,
)
from lottie.serve.service import (
    AgentExecutionError,
    AgentLoadError,
    AgentNotFoundError,
    AgentService,
    InvalidInputError,
)

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

    def _model_not_found(model: str) -> JSONResponse:
        return json_error(
            404, f"model '{model}' not found",
            type_="invalid_request_error", code="model_not_found",
        )

    async def list_models(request: Request) -> JSONResponse:
        from lottie.project.discovery import discover_agents

        created = int(time.time())
        data = [
            {"id": unit.name, "object": "model", "created": created, "owned_by": "lottie"}
            for unit in discover_agents(root)
            if _chat_config(root, unit.name) is not None
        ]
        return JSONResponse({"object": "list", "data": data})

    async def chat_completions(request: Request) -> JSONResponse:
        # 1. parse
        try:
            body = await request.json()
            req = ChatCompletionRequest.model_validate(body)
        except (ValueError, ValidationError):
            return json_error(400, "invalid request body", type_="invalid_request_error")

        # 2. streaming not supported this slice
        if req.stream:
            return json_error(
                400, "streaming is not supported", type_="invalid_request_error"
            )

        # 3. resolve a chat-capable model
        chat = _chat_config(root, req.model)
        if chat is None:
            return _model_not_found(req.model)

        # 4. last user message -> typed payload
        content = last_user_message(req)
        if content is None:
            return json_error(
                400, "no user message in request", type_="invalid_request_error"
            )
        payload = {chat.input_field: content}

        # 5. run through the core (off the event loop)
        try:
            result = await anyio.to_thread.run_sync(
                lambda: svc.run_agent(req.model, payload)
            )
        except InputSecurityViolation:
            return json_error(
                400, "request blocked by content policy",
                type_="invalid_request_error", code="content_filter",
            )
        except OutputSecurityViolation as exc:
            return JSONResponse(
                chat_completion_dict(
                    agent=req.model,
                    content="",
                    input_tokens=exc.input_tokens,
                    output_tokens=exc.output_tokens,
                    latency_ms=0.0,
                    cost_usd=0.0,
                    status="content_filter",
                    finish_reason="content_filter",
                )
            )
        except InvalidInputError:
            return json_error(
                400, f"input does not fit model '{req.model}'", type_="invalid_request_error"
            )
        except AgentNotFoundError:
            return _model_not_found(req.model)
        except (AgentLoadError, AgentExecutionError):
            return json_error(500, "internal error", type_="internal_error")

        # 6. map output -> assistant content
        answer = str(result.output.get(chat.output_field, ""))
        return JSONResponse(
            chat_completion_dict(
                agent=req.model,
                content=answer,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=result.latency_ms,
                cost_usd=result.cost_usd,
                status=result.status,
            )
        )

    app = Starlette(
        routes=[
            Route("/v1/models", list_models, methods=["GET"]),
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        ]
    )
    return app
