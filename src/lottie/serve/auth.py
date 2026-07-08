"""API-key auth middleware for the HTTP transport (minimum-viable production).

Fail-closed WHEN configured, open when not: valid keys come from the env var
``LOTTIE_API_KEYS`` (comma-separated), read per-request so rotation needs no restart and
tests can monkeypatch. Unset/empty -> no auth (dev + back-compat). Accepts either
``Authorization: Bearer <key>`` or ``X-API-Key: <key>``; constant-time comparison; the 401
message never echoes the presented token.

Starlette is imported here only -> keep this module out of ``serve/__init__``.
"""

from __future__ import annotations

import hmac
import os
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from lottie.serve.error_map import json_error

_ENV_KEYS = "LOTTIE_API_KEYS"


def _configured_keys() -> list[str]:
    raw = os.getenv(_ENV_KEYS, "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def _presented_token(request: Request) -> str | None:
    auth = request.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[len("bearer ") :].strip()
    return request.headers.get("x-api-key")


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    """Require a valid API key when LOTTIE_API_KEYS is set; otherwise pass through."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        keys = _configured_keys()
        if not keys:  # unset -> auth disabled (open)
            return await call_next(request)
        token = _presented_token(request)
        if token is not None and any(hmac.compare_digest(token, k) for k in keys):
            return await call_next(request)
        return json_error(
            401,
            "missing or invalid API key",
            type_="invalid_request_error",
            code="unauthorized",
        )
