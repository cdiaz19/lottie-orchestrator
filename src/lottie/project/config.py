"""Project resolution and typed configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

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


class ChatConfig(BaseModel):
    """Opt-in mapping that exposes an agent on the OpenAI chat endpoint."""

    input_field: str   # last user message content -> Input.<input_field>
    output_field: str  # Output.<output_field> -> assistant message content


class RecallConfig(BaseModel):
    """Per-agent recall-injection config. Disabled by default."""

    enabled: bool = False
    limit: int = 5  # top-K semantic notes injected as data context


class ReflectConfig(BaseModel):
    """Per-agent post-run reflection config. Disabled by default."""

    enabled: bool = False


class TrajectoryConfig(BaseModel):
    """Per-agent episodic trajectory persistence. Disabled by default.

    Spends no tokens — unlike reflection this writes no LLM call. Enabling it is what
    gives `lottie reflect` and skill distillation a corpus to read.
    """

    enabled: bool = False
    max_chars: int = 4000  # per-field bound on the raw task/outcome text


class MemoryConfig(BaseModel):
    """Per-agent memory store config. Disabled by default (agent keeps NullMemoryClient)."""

    enabled: bool = False
    backend: Literal["sqlite", "null", "mock"] = "sqlite"
    path: str = ".lottie/memory.db"  # resolved relative to the project root
    namespace: str | None = None  # memory namespace; None → resolved to the agent name
    recall: RecallConfig = RecallConfig()
    reflect: ReflectConfig = ReflectConfig()
    trajectory: TrajectoryConfig = TrajectoryConfig()


class CompactionConfig(BaseModel):
    """Summarise older turns when a run approaches its context window. OFF by default.

    Spends tokens (one LLM call per compaction), so it is counted against the run's
    budget and skipped when that budget is exhausted.
    """

    enabled: bool = False
    max_context_tokens: int = 8000  # approximate; see memory/compaction.py
    keep_recent: int = 6            # most recent turns kept verbatim; must be >= 1


class HarnessConfig(BaseModel):
    """Long-running ergonomics (V2 S5)."""

    compaction: CompactionConfig = CompactionConfig()


class ModuleConfig(BaseModel):
    """Enable or disable one mounted runtime module (V3 S6).

    Only `enabled` for now. Built-in modules keep their existing top-level config keys
    (`budget_usd`, `capabilities`, `memory.*`, ...) rather than growing a second way to
    configure the same thing; this block is for switching a module OFF and, from E7, for
    third-party module settings.
    """

    enabled: bool = True


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: str
    model_params: dict[str, object] = {}
    capabilities: list[str] = []
    policies: list[str] = []
    workers: list[str] = []  # mesh routing allow-set (capability enforcement)
    interrupt_before: list[str] = []  # mesh workers that pause for human approval (HITL)
    budget_usd: float | None = None  # per-agent cumulative spend cap; None = unlimited
    max_run_usd: float | None = None  # per-run cost ceiling + atomic-reservation amount (TOCTOU)
    max_run_tokens: int | None = None  # per-run token cap; None = unlimited
    max_turns: int | None = None  # per-run LLM-completion cap (runaway-loop guard); None = off
    memory: MemoryConfig = MemoryConfig()
    harness: HarnessConfig = HarnessConfig()
    #: name -> {enabled}. Unknown names are rejected by `lottie doctor`, since a typo
    #: here would silently leave a security gate mounted.
    modules: dict[str, ModuleConfig] = {}
    chat: ChatConfig | None = None  # None = agent not exposed on /v1/chat/completions


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
