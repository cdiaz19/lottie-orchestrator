"""Discover and load project units.

Discovery (`discover_*`) reads filesystem + YAML metadata WITHOUT importing user
code, so a broken agent cannot crash `status`/`doctor`. Loading
(`load_agent_class`) imports user code and is used only by `run`.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Literal

import typer
from pydantic import BaseModel

from lottie.core import BaseAgent, SecurityGateProtocol
from lottie.governance.capability import build_capability_gate
from lottie.governance.cost import build_cost_gate
from lottie.governance.policy import build_policy_gate
from lottie.llm import LLMProvider
from lottie.memory.store import build_memory_client
from lottie.project.config import AgentConfig, load_agent_config


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


def _import_unit_module(
    root: Path, dotted: str, name: str, kind: str = "agent"
) -> ModuleType:
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
        raise typer.BadParameter(f"{kind} '{name}' not found") from exc
    except ImportError as exc:
        raise typer.BadParameter(f"{kind} '{name}' failed to import: {exc}") from exc


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


def _find_model(
    module: ModuleType,
    suffix: Literal["Input", "Output"],
    kind: Literal["agent", "skill"],
    name: str,
) -> type[BaseModel]:
    """Return the BaseModel subclass for `suffix` defined in `module`.

    Accepts the legacy bare name (`Input`/`Output`) or the prefixed
    `<ClassName><suffix>` convention produced by the scaffold templates.
    """
    candidate = getattr(module, suffix, None)
    if isinstance(candidate, type) and issubclass(candidate, BaseModel):
        return candidate
    candidates = [
        attr
        for attr in vars(module).values()
        if isinstance(attr, type)
        and issubclass(attr, BaseModel)
        and attr is not BaseModel
        and attr.__name__.endswith(suffix)
        and attr.__module__ == module.__name__
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise typer.BadParameter(
        f"{kind}s/{name}/schema.py must define an `{suffix}(BaseModel)` class"
        f" or exactly one `<Name>{suffix}(BaseModel)` class"
        f" — found {len(candidates)}"
    )


def load_schema_models(
    root: Path, kind: Literal["agent", "skill"], name: str
) -> tuple[type[BaseModel], type[BaseModel]]:
    """Import {kind}s/<name>/schema.py and return its (Input, Output) models."""
    module = _import_unit_module(root, f"{kind}s.{name}.schema", name, kind)
    return _find_model(module, "Input", kind, name), _find_model(module, "Output", kind, name)


def load_input_model(root: Path, name: str) -> type[BaseModel]:
    """Import agents/<name>/schema.py and return its Input model (used by `run`)."""
    return load_schema_models(root, "agent", name)[0]


def load_system_prompt(root: Path, name: str) -> str | None:
    """Return SYSTEM_PROMPT from agents/<name>/prompts.py.

    Returns None only when prompts.py is absent, or when SYSTEM_PROMPT is
    missing / not a str, so `inspect` can render `—`. A prompts.py that exists
    but fails to import propagates the error rather than being masked.
    """
    if not (root / "agents" / name / "prompts.py").is_file():
        return None
    module = _import_unit_module(root, f"agents.{name}.prompts", name)
    prompt = getattr(module, "SYSTEM_PROMPT", None)
    return prompt if isinstance(prompt, str) else None


def instantiate_agent(
    agent_cls: type[BaseAgent[BaseModel, BaseModel]],
    *,
    llm: LLMProvider,
    root: Path,
    config: AgentConfig,
    enable_benchmarks: bool | None = None,
    security_gate: SecurityGateProtocol | None = None,
) -> BaseAgent[BaseModel, BaseModel]:
    """Construct an agent using the from_project seam when available.

    Uses ``agent_cls.from_project(llm=llm, root=root, config=config)`` for
    knowledge-backed agents that define the classmethod, and falls back to the
    plain ``agent_cls(llm=llm)`` constructor for simple agents.  This is the
    single canonical dispatch point — both the CLI (``lottie run``) and the
    serving core (``AgentService``) delegate here so the logic is not duplicated.

    Parameters
    ----------
    enable_benchmarks:
        When ``False`` the constructed agent (and any nested skills) will NOT
        write benchmark metric records during ``run()``.  Pass ``False`` from
        the benchmark runner so eval cases don't produce spurious nested writes.
        ``None`` (default) defers to the ``LOTTIE_DISABLE_BENCHMARKS`` env var,
        which is the correct behaviour for the ``serve`` and ``run`` paths.
    """
    if hasattr(agent_cls, "from_project"):
        agent: BaseAgent[BaseModel, BaseModel] = agent_cls.from_project(
            llm=llm, root=root, config=config, enable_benchmarks=enable_benchmarks
        )
    else:
        agent = agent_cls(llm=llm, enable_benchmarks=enable_benchmarks)
    agent.set_policy(
        build_policy_gate(root, policies=config.policies, capabilities=config.capabilities)
    )
    # Budget reads the audit ledger under `root`; the agent's audit writes under its
    # benchmarks_root (default cwd). Both resolve to the project root for `lottie run`
    # and `serve` (cwd == root). A caller that instantiates with root != cwd would split
    # the ledger — out of scope (no shipped call site does this).
    agent.set_cost_gate(
        build_cost_gate(
            root,
            agent=agent.name,
            budget_usd=config.budget_usd,
            max_run_usd=config.max_run_usd,
        )
    )
    agent.set_run_limits(
        max_run_tokens=config.max_run_tokens, max_turns=config.max_turns
    )
    # Rule 11: per-skill-call whitelist. Empty capabilities -> NullCapabilityGate
    # (no enforcement); a non-empty list blocks any skill the agent didn't declare.
    agent.set_capability_gate(build_capability_gate(capabilities=config.capabilities))
    # V2 S0: attach a persistent memory client when the agent opts in. Disabled →
    # the agent keeps its default NullMemoryClient (fail-loud on use).
    if config.memory.enabled:
        agent.set_memory(
            build_memory_client(
                root, backend=config.memory.backend, path=config.memory.path
            )
        )
        if config.memory.recall.enabled:
            agent.set_recall(
                enabled=True,
                namespace=config.memory.namespace or agent.name,
                limit=config.memory.recall.limit,
            )
        if config.memory.reflect.enabled:
            if config.max_run_tokens is None:
                warnings.warn(
                    f"agent {agent.name!r}: memory.reflect is enabled without max_run_tokens — "
                    "reflection LLM spend is unbounded per run. Set max_run_tokens to bound it.",
                    stacklevel=2,
                )
            agent.set_reflect(
                enabled=True,
                namespace=config.memory.namespace or agent.name,
            )
        # V2 S3a: append each run to episodic memory. Spends no tokens (no LLM call), so
        # unlike reflect it needs no max_run_tokens warning. This is what gives
        # `lottie reflect` and S3b distillation a corpus to read.
        if config.memory.trajectory.enabled:
            agent.set_trajectory(
                enabled=True,
                namespace=config.memory.namespace or agent.name,
                max_chars=config.memory.trajectory.max_chars,
            )
    # V2 S5a: context compaction for long runs. Independent of memory.enabled — a run
    # can outgrow its window whether or not the agent learns.
    if config.harness.compaction.enabled:
        agent.set_compaction(
            enabled=True,
            max_context_tokens=config.harness.compaction.max_context_tokens,
            keep_recent=config.harness.compaction.keep_recent,
        )
    # V3 S6: the `modules:` block switches mounted modules off. A disabled module is
    # never constructed, so it costs nothing at run time.
    disabled = frozenset(
        name for name, module in config.modules.items() if not module.enabled
    )
    if disabled:
        agent.set_disabled_modules(disabled)
    # Rules 8 & 9: input/output security gate on the BaseAgent chokepoint. Attached only
    # when a caller injects one (the CLI passes serve.security.SecurityGate); serve leaves
    # it None and gates externally in AgentService, so no path is gated twice.
    if security_gate is not None:
        agent.set_security_gate(security_gate)
    return agent


def required_fields(model: type[BaseModel]) -> list[str]:
    """Field names on `model` that have no default (must be supplied)."""
    return [
        field_name
        for field_name, field in model.model_fields.items()
        if field.is_required()
    ]
