"""Discover and load project units.

Discovery (`discover_*`) reads filesystem + YAML metadata WITHOUT importing user
code, so a broken agent cannot crash `status`/`doctor`. Loading
(`load_agent_class`) imports user code and is used only by `run`.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Literal

import typer
from pydantic import BaseModel

from lottie.core import BaseAgent
from lottie.project.config import load_agent_config


class UnitInfo(BaseModel):
    name: str
    kind: Literal["agent", "skill"]
    provider: str | None
    path: Path


def discover_agents(root: Path) -> list[UnitInfo]:
    return _discover(root / "agents", "agent", "agent.py")


def discover_skills(root: Path) -> list[UnitInfo]:
    return _discover(root / "skills", "skill", "skill.py")


def _discover(base: Path, kind: Literal["agent", "skill"], entry: str) -> list[UnitInfo]:
    if not base.is_dir():
        return []
    units: list[UnitInfo] = []
    for unit_dir in sorted(base.iterdir()):
        if not unit_dir.is_dir() or not (unit_dir / entry).is_file():
            continue
        provider = _provider_of(unit_dir) if kind == "agent" else None
        units.append(UnitInfo(name=unit_dir.name, kind=kind, provider=provider, path=unit_dir))
    return units


def _provider_of(unit_dir: Path) -> str | None:
    try:
        return load_agent_config(unit_dir).provider
    except Exception:  # noqa: BLE001
        return None


# Tracks which project root each top-level package (`agents`/`skills`) was last
# imported from, so same-root calls reuse the cache (one set of class objects)
# while a new root forces a fresh re-import.
_loaded_roots: dict[str, str] = {}


def _import_unit_module(root: Path, dotted: str, name: str) -> ModuleType:
    """Import a dotted module from a project root.

    The first import of a top-level package (`agents`/`skills`) from a given root
    purges any stale cache for that package, so a different project — or a re-run
    in the same process — imports the right code. Imports from the SAME root reuse
    the cache, so a module and its relative imports share one set of class objects.

    Caveat: the cache keys on root path only, not content. A long-lived host that
    edits a unit file in place and re-imports (hot-reload) would get the stale
    module — fine for the per-process CLI; revisit (content signature / reset) for
    a future `serve`/MCP host.
    """
    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)
    top = dotted.split(".")[0]
    if _loaded_roots.get(top) != root_str:
        for cached in [m for m in list(sys.modules) if m == top or m.startswith(f"{top}.")]:
            del sys.modules[cached]
        importlib.invalidate_caches()
        _loaded_roots[top] = root_str
    try:
        return importlib.import_module(dotted)
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(f"agent '{name}' not found") from exc
    except ImportError as exc:
        raise typer.BadParameter(f"agent '{name}' failed to import: {exc}") from exc


def load_agent_class(root: Path, name: str) -> type[BaseAgent[BaseModel, BaseModel]]:
    """Import agents/<name>/agent.py and return its single BaseAgent subclass.

    Only agents are loaded dynamically; skills are stateless and need no import.
    """
    module = _import_unit_module(root, f"agents.{name}.agent", name)
    classes: list[type[BaseAgent[BaseModel, BaseModel]]] = []
    for attr in vars(module).values():
        if (
            isinstance(attr, type)
            and issubclass(attr, BaseAgent)
            and attr is not BaseAgent
            and attr.__module__ == module.__name__
        ):
            classes.append(attr)
    if len(classes) != 1:
        raise typer.BadParameter(
            f"expected exactly one agent class in agents/{name}/agent.py,"
            f" found {len(classes)}"
        )
    return classes[0]


def load_input_model(root: Path, name: str) -> type[BaseModel]:
    """Import agents/<name>/schema.py and return its `Input` model."""
    module = _import_unit_module(root, f"agents.{name}.schema", name)
    candidate = getattr(module, "Input", None)
    if not (isinstance(candidate, type) and issubclass(candidate, BaseModel)):
        raise typer.BadParameter(
            f"agents/{name}/schema.py must define an `Input(BaseModel)` class"
        )
    return candidate


def required_fields(model: type[BaseModel]) -> list[str]:
    """Field names on `model` that have no default (must be supplied)."""
    return [
        field_name
        for field_name, field in model.model_fields.items()
        if field.is_required()
    ]
