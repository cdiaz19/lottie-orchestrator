"""limit/offset pagination for the HTTP list endpoints. Starlette-free (takes a plain
mapping of query params) so it is unit-testable without a request."""

from __future__ import annotations

from collections.abc import Mapping

MAX_LIMIT = 100  # default page size AND the ceiling for an explicit limit


def page_bounds(query: Mapping[str, str]) -> tuple[int, int]:
    """Resolve (limit, offset) from query params. Garbage or absent -> defaults; limit is
    clamped to 1..MAX_LIMIT and offset to >= 0."""
    limit = _int_or(query.get("limit"), MAX_LIMIT)
    offset = _int_or(query.get("offset"), 0)
    limit = max(1, min(limit, MAX_LIMIT))
    offset = max(0, offset)
    return limit, offset


def _int_or(raw: str | None, default: int) -> int:
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default
