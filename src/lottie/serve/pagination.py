"""limit/offset pagination for the HTTP list endpoints. Starlette-free (takes a plain
mapping of query params) so it is unit-testable without a request."""

from __future__ import annotations

from collections.abc import Mapping

MAX_LIMIT = 100  # ceiling for an EXPLICIT limit param


def page_bounds(query: Mapping[str, str]) -> tuple[int | None, int]:
    """Resolve (limit, offset) from query params.

    `limit` absent (or non-integer) -> None, meaning **return everything** — so an
    un-paginated caller never silently gets a truncated list. An explicit `limit` is clamped
    to 1..MAX_LIMIT. `offset` -> >= 0 (garbage -> 0).
    """
    raw_limit = query.get("limit")
    limit: int | None = None
    if raw_limit is not None:
        parsed = _int_or(raw_limit, None)
        if parsed is not None:
            limit = max(1, min(parsed, MAX_LIMIT))
    offset = max(0, _int_or(query.get("offset"), 0) or 0)
    return limit, offset


def slice_page[T](items: list[T], limit: int | None, offset: int) -> list[T]:
    """Apply (limit, offset); limit None -> from offset to the end (no cap)."""
    return items[offset:] if limit is None else items[offset : offset + limit]


def _int_or(raw: str | None, default: int | None) -> int | None:
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default
