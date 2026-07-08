"""Per-identity token-bucket rate limiting for the HTTP transport.

Cap from env ``LOTTIE_RATE_LIMIT_PER_MIN`` (int), read per-request. Unset/0 -> no limit.
Identity = the presented API key if any, else the client host. In-memory buckets, so this is
**per-process** (not distributed) — honest minimum-viable production. Over the cap -> 429.

Starlette is imported here only -> keep this module out of ``serve/__init__``.
"""

from __future__ import annotations

import os
import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from lottie.serve.error_map import json_error

_ENV_RATE = "LOTTIE_RATE_LIMIT_PER_MIN"


def _rate_per_min() -> int:
    try:
        return max(0, int(os.getenv(_ENV_RATE, "0")))
    except (TypeError, ValueError):
        return 0


def _identity(request: Request) -> str:
    # Identity is the PRESENTED (unvalidated) key or the client host. Known V1 limitations:
    # with auth OFF a client can rotate the key header for fresh buckets, and with auth ON a
    # flood of bogus keys grows the bucket map (no eviction). Acceptable for the per-process,
    # minimum-viable limiter; a validated-identity + LRU/TTL sweep is a follow-up.
    auth = request.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return "k:" + auth[len("bearer ") :].strip()
    key = request.headers.get("x-api-key")
    if key:
        return "k:" + key
    return "h:" + (request.client.host if request.client else "unknown")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token-bucket limiter: capacity = rate, refill = rate/60 per second, per identity."""

    def __init__(self, app: object) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        # identity -> (tokens, last_refill_monotonic)
        self._buckets: dict[str, tuple[float, float]] = {}

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rate = _rate_per_min()
        if rate <= 0:  # disabled
            return await call_next(request)
        now = time.monotonic()
        ident = _identity(request)
        tokens, last = self._buckets.get(ident, (float(rate), now))
        tokens = min(float(rate), tokens + (now - last) * rate / 60.0)
        if tokens < 1.0:
            self._buckets[ident] = (tokens, now)
            return json_error(
                429, "rate limit exceeded", type_="rate_limit_error", code="rate_limited"
            )
        self._buckets[ident] = (tokens - 1.0, now)
        return await call_next(request)
