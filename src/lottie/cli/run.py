"""`lottie run <name>` — load and execute an agent end-to-end."""

from __future__ import annotations

from typing import Annotated

import typer
from pydantic import BaseModel, ValidationError

from lottie.llm import build_provider
from lottie.project.config import find_project_root, load_agent_config
from lottie.project.discovery import load_agent_class, load_input_model, required_fields


def run(
    name: str,
    input_json: Annotated[
        str | None, typer.Option("--input", help="JSON input payload for the agent.")
    ] = None,
    provider: Annotated[
        str | None, typer.Option("--provider", help="Override the LLM provider.")
    ] = None,
) -> None:
    """Run an agent, printing its output as JSON."""
    root = find_project_root()
    unit_dir = root / "agents" / name
    if not (unit_dir / "agent.py").is_file():
        raise typer.BadParameter(f"agent '{name}' not found")

    cfg = load_agent_config(unit_dir)
    llm = build_provider(provider or cfg.provider)
    input_model = load_input_model(root, name)
    data = _build_input(input_model, input_json, name)

    agent_cls = load_agent_class(root, name)
    if hasattr(agent_cls, "from_project"):
        agent = agent_cls.from_project(llm=llm, root=root, config=cfg)
    else:
        agent = agent_cls(llm=llm)
    try:
        result = agent.run(data)
    except typer.Exit:
        raise
    except Exception as exc:
        raise typer.BadParameter(f"agent '{name}' failed: {exc}") from exc
    typer.echo(result.model_dump_json(indent=2))


def _build_input(
    input_model: type[BaseModel], input_json: str | None, name: str
) -> BaseModel:
    if input_json is not None:
        try:
            return input_model.model_validate_json(input_json)
        except ValidationError as exc:
            raise typer.BadParameter(f"invalid --input for '{name}': {exc}") from exc
    missing = required_fields(input_model)
    if missing:
        raise typer.BadParameter(
            f"agent '{name}' needs --input with fields: {', '.join(missing)}"
        )
    return input_model()
