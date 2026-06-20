"""OpenAI-compatible HTTP transport over AgentService.

Pure wrapper, same shape as serve/mcp_server.py. Imports Starlette at module top,
so it is imported lazily (never from serve/__init__.py) — the base install needs
neither [serve] nor [api].
"""

from __future__ import annotations

import contextvars
import logging
import time
from collections.abc import AsyncIterator, Generator
from pathlib import Path
from typing import cast

import anyio
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from lottie.project.config import ChatConfig, load_agent_config
from lottie.serve.error_map import json_error
from lottie.serve.errors import InputSecurityViolation, OutputSecurityViolation
from lottie.serve.openai_schema import (
    ChatChunkEncoder,
    ChatCompletionRequest,
    chat_completion_chunks,
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

# Sentinel: StopIteration cannot cross the to_thread/await boundary as itself.
_STREAM_DONE = object()


def _safe_next(gen: Generator[str, None, None]) -> str | object:
    try:
        return next(gen)
    except StopIteration:
        return _STREAM_DONE


async def _sse_real(model: str, gen: Generator[str, None, None]) -> AsyncIterator[str]:
    """Bridge a sync (run_stream -> scan_stream) generator to SSE, pulling each delta off
    the event loop.

    A secret trips scan_stream -> OutputSecurityViolation -> finish 'content_filter' (the
    secret line is never yielded); a policy/cost/agent failure -> 'error'. `gen.close()` on
    exit drives run_stream's PARTIAL audit on early client disconnect.

    run_stream sets/resets the `_audit_depth` ContextVar token ACROSS yields, so every pull
    must run in the SAME context. anyio.to_thread copies a fresh context per call, which would
    make the final reset() raise "Token created in a different Context". We pin ONE context and
    drive every next()/close() through `ctx.run`, so the set and reset pair cleanly.
    """
    enc = ChatChunkEncoder(model)
    yield enc.role()
    ctx = contextvars.copy_context()
    finish = "stop"
    try:
        while True:
            item = await anyio.to_thread.run_sync(ctx.run, _safe_next, gen)
            if item is _STREAM_DONE:
                break
            assert isinstance(item, str)  # narrow sentinel for mypy; scan_stream yields str
            yield enc.content(item)
    except OutputSecurityViolation:
        finish = "content_filter"
    except Exception:  # noqa: BLE001 — policy/cost/agent failure -> terminal error finish
        # the audit row is written by run_stream; log too so the failure is visible in server logs
        logger.warning("mid-stream failure for model %r", model, exc_info=True)
        finish = "error"
    finally:
        ctx.run(gen.close)  # close in the pinned context too, so run_stream's reset() pairs
    yield enc.finish(finish)
    yield enc.done()


def _chat_config(root: Path, name: str) -> ChatConfig | None:
    """The agent's chat block, or None if the agent is missing/unloadable/not chat."""
    if not (root / "agents" / name / "agent.py").is_file():
        return None
    try:
        return load_agent_config(root / "agents" / name).chat
    except Exception as exc:  # noqa: BLE001 — a broken agent is simply not chat-exposed
        logger.warning("skipping agent %r for chat: %s", name, exc)
        return None


def openai_routes(svc: AgentService, root: Path) -> list[Route]:
    """The OpenAI-compat routes (/v1/models, /v1/chat/completions), closed over svc + root."""

    def _model_not_found(model: str) -> JSONResponse:
        return json_error(
            404, f"model '{model}' not found",
            type_="invalid_request_error", code="model_not_found",
        )

    def _stream_response(agent: str, content: str, finish_reason: str) -> StreamingResponse:
        """SSE response of a completed chat result — role/content/finish chunks + [DONE]."""
        return StreamingResponse(
            iter(chat_completion_chunks(agent=agent, content=content, finish_reason=finish_reason)),
            media_type="text/event-stream",
        )

    async def _real_stream_or_none(model: str, payload: dict[str, str]) -> Response | None:
        """An SSE StreamingResponse for an opt-in agent; None if not streamable (caller falls back);
        a JSON error Response for a pre-stream failure (input/validation/load)."""
        try:
            gen = await anyio.to_thread.run_sync(lambda: svc.stream_agent(model, payload))
        except InputSecurityViolation:
            return json_error(
                400, "request blocked by content policy",
                type_="invalid_request_error", code="content_filter",
            )
        except InvalidInputError:
            return json_error(
                400, f"input does not fit model '{model}'", type_="invalid_request_error"
            )
        except AgentNotFoundError:
            return _model_not_found(model)
        except (AgentLoadError, AgentExecutionError):
            return json_error(500, "internal error", type_="internal_error")
        if gen is None:
            return None
        # scan_stream returns an Iterator[str] that is always a generator at runtime;
        # cast to Generator so _sse_real can call .close() for cancellation/cleanup.
        return StreamingResponse(
            _sse_real(model, cast(Generator[str, None, None], gen)),
            media_type="text/event-stream",
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

    async def chat_completions(request: Request) -> Response:
        # 1. parse
        try:
            body = await request.json()
            req = ChatCompletionRequest.model_validate(body)
        except (ValueError, ValidationError):
            return json_error(400, "invalid request body", type_="invalid_request_error")

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

        # 5a. real-token streaming for opt-in agents; None -> fall through to format-level
        if req.stream:
            streamed = await _real_stream_or_none(req.model, payload)
            if streamed is not None:
                return streamed

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
            if req.stream:
                return _stream_response(req.model, "", "content_filter")
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
        if req.stream:
            return _stream_response(req.model, answer, "stop")
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

    return [
        Route("/v1/models", list_models, methods=["GET"]),
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
    ]


def build_openai_app(root: Path, *, service: AgentService | None = None) -> Starlette:
    """Build a Starlette app exposing chat-capable agents over the OpenAI API."""
    svc = service or AgentService(root)
    return Starlette(routes=openai_routes(svc, root))
