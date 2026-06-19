"""The combined HTTP app `lottie serve --port` serves: OpenAI-compat + Lottie-native REST,
over ONE AgentService (one chokepoint -> shared security + audit/policy/cost).

Imports Starlette at module top -> lazy-only; never imported from serve/__init__.py."""

from __future__ import annotations

from pathlib import Path

from starlette.applications import Starlette

from lottie.serve.openai_app import openai_routes
from lottie.serve.rest_app import rest_routes
from lottie.serve.service import AgentService


def build_http_app(root: Path) -> Starlette:
    """Build the Starlette app exposing both the OpenAI and REST route groups."""
    svc = AgentService(root)
    return Starlette(routes=[*openai_routes(svc, root), *rest_routes(svc, root)])
