"""MCP stdio transport: expose each agent as a typed MCP tool.

Pure wrapper over AgentService. Imports the `mcp` SDK at module top, so this
module is imported lazily (never from serve/__init__.py) — the base install
does not require the optional [serve] extra.
"""

from __future__ import annotations

import logging
from pathlib import Path

from mcp import types
from mcp.server.lowlevel import Server

from lottie.project.discovery import (
    discover_agents,
    load_input_model,
    load_system_prompt,
)
from lottie.serve.service import AgentService

logger = logging.getLogger(__name__)


def _tool_description(root: Path, name: str) -> str:
    """First line of the agent's system prompt, or a generic fallback."""
    prompt = load_system_prompt(root, name)
    if prompt and prompt.strip():
        return prompt.strip().splitlines()[0]
    return f"Run the {name} agent."


def build_mcp_server(root: Path, *, service: AgentService | None = None) -> Server:
    """Build an MCP Server exposing one typed tool per healthy agent under `root`."""
    svc = service or AgentService(root)
    _ = svc  # reserved for call_tool (Task 3)

    tools: dict[str, types.Tool] = {}
    for unit in discover_agents(root):
        try:
            input_model = load_input_model(root, unit.name)
            description = _tool_description(root, unit.name)
        except Exception as exc:  # noqa: BLE001 — a broken agent is skipped, not fatal
            logger.warning("skipping agent %r: %s", unit.name, exc)
            continue
        tools[unit.name] = types.Tool(
            name=unit.name,
            description=description,
            inputSchema=input_model.model_json_schema(),
        )

    server: Server = Server("lottie")

    @server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
    async def _list_tools() -> list[types.Tool]:
        return list(tools.values())

    return server
