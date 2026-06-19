"""Map an error into an OpenAI-shaped JSONResponse. Kept separate from openai_app
so it is shared by both transports (the OpenAI routes and the REST routes)."""

from __future__ import annotations

from starlette.responses import JSONResponse

from lottie.serve.openai_schema import error_dict


def json_error(
    status: int, message: str, *, type_: str, code: str | None = None
) -> JSONResponse:
    """An OpenAI error envelope at the given HTTP status. `message` carries no payload."""
    return JSONResponse(error_dict(message, type_=type_, code=code), status_code=status)
