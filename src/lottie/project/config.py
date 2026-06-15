"""Project resolution and typed configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError


class Providers(BaseModel):
    default: str
    fallback: str | None = None


class Registry(BaseModel):
    agents: str = "agents/"
    skills: str = "skills/"


class LottieConfig(BaseModel):
    project: str
    providers: Providers
    policies: list[str] = []
    registry: Registry = Registry()


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: str
    model_params: dict[str, object] = {}
    capabilities: list[str] = []
    policies: list[str] = []
    workers: list[str] = []  # mesh routing allow-set (capability enforcement)
    interrupt_before: list[str] = []  # mesh workers that pause for human approval (HITL)
    budget_usd: float | None = None  # per-agent cumulative spend cap; None = unlimited


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from `start` (default cwd) to the dir containing lottie.yaml."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "lottie.yaml").is_file():
            return candidate
    raise typer.BadParameter("not a Lottie project — run `lottie init` first.")


def load_lottie_config(root: Path) -> LottieConfig:
    return _load_yaml_model(root / "lottie.yaml", LottieConfig)


def load_agent_config(unit_dir: Path) -> AgentConfig:
    return _load_yaml_model(unit_dir / "config.yaml", AgentConfig)


def _load_yaml_model[M: BaseModel](path: Path, model: type[M]) -> M:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise typer.BadParameter(f"cannot read {path}: {exc}") from exc
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise typer.BadParameter(f"invalid {path.name}: {exc}") from exc
