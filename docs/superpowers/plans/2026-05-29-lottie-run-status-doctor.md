# `lottie run` / `status` / `doctor` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `lottie run <name>` (execute an agent end-to-end), `lottie status` (inspect project + registry), and `lottie doctor` (environment health), built on a shared `src/lottie/project/` layer.

**Architecture:** New `lottie.project` package owns project resolution, typed config (`LottieConfig`/`AgentConfig` via YAML), and unit discovery. Discovery is split: metadata `discover_*` (filesystem + YAML, no import) for `status`; `load_agent_class` (imports user code) for `run`. Thin `cli/{run,status,doctor}.py` delegate in. A `lottie.llm.build_provider` factory is the single provider construction point; tests monkeypatch `litellm.completion` beneath it.

**Tech Stack:** Python 3.12, Typer + `CliRunner` + pytest (no real LLM), Pydantic v2, PyYAML, rich, uv.

**Spec:** `docs/superpowers/specs/2026-05-29-lottie-run-status-doctor-design.md`

---

## File structure

| File | Responsibility |
|---|---|
| `pyproject.toml` (modify) | add `pyyaml` dep + `types-PyYAML` dev dep |
| `src/lottie/llm/__init__.py` (modify) | add + export `build_provider` |
| `src/lottie/llm/tests/test_llm_base.py` (modify) | test for `build_provider` |
| `src/lottie/project/__init__.py` (create) | package marker |
| `src/lottie/project/config.py` (create) | root resolution + typed configs + YAML load |
| `src/lottie/project/discovery.py` (create) | discover (no import) + load (import) units |
| `src/lottie/project/tests/__init__.py` (create) | test package marker |
| `src/lottie/project/tests/test_config.py` (create) | config + root tests |
| `src/lottie/project/tests/test_discovery.py` (create) | discovery + load tests |
| `src/lottie/cli/status.py` (create) | `status` command |
| `src/lottie/cli/doctor.py` (create) | `doctor` command |
| `src/lottie/cli/run.py` (create) | `run` command |
| `src/lottie/cli/app.py` (modify) | register the three commands |
| `src/lottie/cli/tests/test_status.py` (create) | status CLI tests |
| `src/lottie/cli/tests/test_doctor.py` (create) | doctor CLI tests |
| `src/lottie/cli/tests/test_run.py` (create) | run CLI tests (monkeypatched LLM) |

---

## Task 1: Dependencies + `build_provider` factory

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/lottie/llm/__init__.py`
- Test: `src/lottie/llm/tests/test_llm_base.py`

- [ ] **Step 1: Add dependencies**

Run: `uv add pyyaml` then `uv add --dev types-pyyaml`
Expected: `pyyaml` in `[project].dependencies`; `types-pyyaml` in the dev group; `uv.lock` updates.

- [ ] **Step 2: Write the failing test**

Append to `src/lottie/llm/tests/test_llm_base.py`:

```python
def test_build_provider_returns_litellm_provider() -> None:
    from lottie.llm import LiteLLMProvider, build_provider

    provider = build_provider("anthropic/claude-sonnet-4-6")
    assert isinstance(provider, LiteLLMProvider)
    assert provider.model == "anthropic/claude-sonnet-4-6"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest src/lottie/llm/tests/test_llm_base.py::test_build_provider_returns_litellm_provider -v`
Expected: FAIL — `ImportError: cannot import name 'build_provider'`.

- [ ] **Step 4: Implement**

Edit `src/lottie/llm/__init__.py`. Add the factory after the existing imports and extend `__all__`:

```python
from lottie.llm.base import LLMProvider, LLMResponse, Message, Role, TokenUsage
from lottie.llm.litellm_provider import LiteLLMProvider
from lottie.llm.mock import MockLLMProvider


def build_provider(model: str) -> LLMProvider:
    """Construct the default LLM provider for a model id.

    Single construction point used by the CLI; tests monkeypatch
    ``litellm.completion`` beneath the returned provider.
    """
    return LiteLLMProvider(model)


__all__ = [
    "LLMProvider",
    "LLMResponse",
    "LiteLLMProvider",
    "Message",
    "MockLLMProvider",
    "Role",
    "TokenUsage",
    "build_provider",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest src/lottie/llm/tests/test_llm_base.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/lottie/llm/__init__.py src/lottie/llm/tests/test_llm_base.py
git commit -m "feat(llm): add build_provider factory; add pyyaml dep"
```

---

## Task 2: `project/config.py` — root resolution + typed config

**Files:**
- Create: `src/lottie/project/__init__.py`
- Create: `src/lottie/project/config.py`
- Create: `src/lottie/project/tests/__init__.py`
- Test: `src/lottie/project/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `src/lottie/project/__init__.py`:

```python
"""Project layer — resolution, typed config, and unit discovery."""
```

Create `src/lottie/project/tests/__init__.py` (empty file).

Create `src/lottie/project/tests/test_config.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
import typer

from lottie.project.config import (
    AgentConfig,
    LottieConfig,
    find_project_root,
    load_agent_config,
    load_lottie_config,
)

_LOTTIE_YAML = """\
project: demo
providers:
  default: anthropic/claude-sonnet-4-6
  fallback: openai/gpt-4o
policies:
  - base
registry:
  agents: agents/
  skills: skills/
"""

_AGENT_YAML = """\
provider: anthropic/claude-sonnet-4-6
model_params:
  temperature: 0.3
capabilities: []
policies:
  - base
memory:
  enabled: false
  namespace: researcher
"""


def test_find_project_root_walks_up(tmp_path: Path) -> None:
    (tmp_path / "lottie.yaml").write_text(_LOTTIE_YAML, encoding="utf-8")
    nested = tmp_path / "agents" / "researcher"
    nested.mkdir(parents=True)
    assert find_project_root(nested) == tmp_path


def test_find_project_root_raises_when_absent(tmp_path: Path) -> None:
    with pytest.raises(typer.BadParameter):
        find_project_root(tmp_path)


def test_load_lottie_config(tmp_path: Path) -> None:
    (tmp_path / "lottie.yaml").write_text(_LOTTIE_YAML, encoding="utf-8")
    cfg = load_lottie_config(tmp_path)
    assert isinstance(cfg, LottieConfig)
    assert cfg.project == "demo"
    assert cfg.providers.default == "anthropic/claude-sonnet-4-6"
    assert cfg.providers.fallback == "openai/gpt-4o"
    assert cfg.policies == ["base"]


def test_load_agent_config_ignores_extra(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(_AGENT_YAML, encoding="utf-8")
    cfg = load_agent_config(tmp_path)
    assert isinstance(cfg, AgentConfig)
    assert cfg.provider == "anthropic/claude-sonnet-4-6"
    assert cfg.model_params == {"temperature": 0.3}
    # `memory` is not a field — extra='ignore' must not raise.


def test_load_lottie_config_malformed_raises(tmp_path: Path) -> None:
    (tmp_path / "lottie.yaml").write_text("project: [unclosed", encoding="utf-8")
    with pytest.raises(typer.BadParameter):
        load_lottie_config(tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/project/tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.project.config'`.

- [ ] **Step 3: Implement**

Create `src/lottie/project/config.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/project/tests/test_config.py -v`
Expected: PASS (5 tests).
Then: `uv run mypy --strict src/lottie/project` and `uv run ruff check src/lottie/project` — clean.

- [ ] **Step 5: Commit**

```bash
git add src/lottie/project/__init__.py src/lottie/project/config.py src/lottie/project/tests
git commit -m "feat(project): add project resolution and typed config loading"
```

---

## Task 3: `project/discovery.py` — metadata discovery (no import)

**Files:**
- Create: `src/lottie/project/discovery.py`
- Test: `src/lottie/project/tests/test_discovery.py`

- [ ] **Step 1: Write the failing test**

Create `src/lottie/project/tests/test_discovery.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.cli import app
from lottie.project.discovery import discover_agents, discover_skills

runner = CliRunner()


def _scaffold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "researcher"]).exit_code == 0
    assert runner.invoke(app, ["create", "skill", "cleaner"]).exit_code == 0
    return demo


def test_discover_agents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    agents = discover_agents(demo)
    assert [a.name for a in agents] == ["researcher"]
    assert agents[0].kind == "agent"
    assert agents[0].provider == "anthropic/claude-sonnet-4-6"


def test_discover_skills(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    skills = discover_skills(demo)
    assert [s.name for s in skills] == ["cleaner"]
    assert skills[0].kind == "skill"
    assert skills[0].provider is None


def test_discover_empty_when_no_dir(tmp_path: Path) -> None:
    assert discover_agents(tmp_path) == []
    assert discover_skills(tmp_path) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/project/tests/test_discovery.py -v`
Expected: FAIL — `ImportError: cannot import name 'discover_agents'`.

- [ ] **Step 3: Implement**

Create `src/lottie/project/discovery.py`:

```python
"""Discover and load project units.

Discovery (`discover_*`) reads filesystem + YAML metadata WITHOUT importing user
code, so a broken agent cannot crash `status`/`doctor`. Loading
(`load_agent_class`) imports user code and is used only by `run`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

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
    except Exception:
        return None
```

(`_provider_of` swallows any config error to `None` — discovery must never raise on a malformed
unit; `status` shows it with a blank provider.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/project/tests/test_discovery.py -v`
Expected: PASS (3 tests).
Then: `uv run mypy --strict src/lottie/project` and `uv run ruff check src/lottie/project` — clean.

- [ ] **Step 5: Commit**

```bash
git add src/lottie/project/discovery.py src/lottie/project/tests/test_discovery.py
git commit -m "feat(project): add unit metadata discovery"
```

---

## Task 4: `project/discovery.py` — load agent class + input model

**Files:**
- Modify: `src/lottie/project/discovery.py`
- Test: `src/lottie/project/tests/test_discovery.py`

- [ ] **Step 1: Write the failing test**

Append to `src/lottie/project/tests/test_discovery.py`:

```python
from lottie.core import BaseAgent
from lottie.project.discovery import load_agent_class, load_input_model, required_fields


def test_load_agent_class(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    cls = load_agent_class(demo, "researcher")
    assert issubclass(cls, BaseAgent)
    assert cls.__name__ == "ResearcherAgent"


def test_load_agent_class_zero_subclasses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    # Overwrite agent.py with no BaseAgent subclass.
    (demo / "agents" / "researcher" / "agent.py").write_text(
        "x = 1\n", encoding="utf-8"
    )
    import typer

    with pytest.raises(typer.BadParameter):
        load_agent_class(demo, "researcher")


def test_load_agent_class_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    import typer

    with pytest.raises(typer.BadParameter):
        load_agent_class(demo, "nope")


def test_load_input_model_and_required_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    model = load_input_model(demo, "researcher")
    assert required_fields(model) == ["query"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/project/tests/test_discovery.py -k "load_ or required" -v`
Expected: FAIL — `ImportError: cannot import name 'load_agent_class'`.

- [ ] **Step 3: Implement**

Add these imports to the top of `src/lottie/project/discovery.py` (beside the existing ones):

```python
import importlib
import sys
from types import ModuleType

import typer

from lottie.core import BaseAgent
```

Append to the bottom of `src/lottie/project/discovery.py`:

```python
def load_agent_class(root: Path, name: str) -> type[BaseAgent]:
    """Import agents/<name>/agent.py and return its single BaseAgent subclass."""
    module = _import_unit_module(root, f"agents.{name}.agent", name)
    classes: list[type[BaseAgent]] = []
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
            f"expected exactly one agent class in agents/{name}/agent.py, found {len(classes)}"
        )
    return classes[0]


def load_input_model(root: Path, name: str) -> type[BaseModel]:
    """Import agents/<name>/schema.py and return its `Input` model."""
    module = _import_unit_module(root, f"agents.{name}.schema", name)
    input_model: type[BaseModel] = module.Input
    return input_model


def required_fields(model: type[BaseModel]) -> list[str]:
    """Field names on `model` that have no default (must be supplied)."""
    return [field_name for field_name, field in model.model_fields.items() if field.is_required()]


def _import_unit_module(root: Path, dotted: str, name: str) -> ModuleType:
    """Import a dotted module from a project root, fresh each call.

    Drops cached `agents`/`skills` modules and re-prioritises `root` on sys.path so
    a different project (or re-run in the same process) imports the right code.
    """
    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)
    for cached in [m for m in list(sys.modules) if m.split(".")[0] in {"agents", "skills"}]:
        del sys.modules[cached]
    importlib.invalidate_caches()
    try:
        return importlib.import_module(dotted)
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(f"agent '{name}' not found") from exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/project/tests/test_discovery.py -v`
Expected: PASS (all discovery tests green).
Then: `uv run mypy --strict src/lottie/project` and `uv run ruff check src/lottie/project` — clean.

- [ ] **Step 5: Commit**

```bash
git add src/lottie/project/discovery.py src/lottie/project/tests/test_discovery.py
git commit -m "feat(project): load agent class and input model for run"
```

---

## Task 5: `lottie status`

**Files:**
- Create: `src/lottie/cli/status.py`
- Modify: `src/lottie/cli/app.py`
- Test: `src/lottie/cli/tests/test_status.py`

- [ ] **Step 1: Write the failing test**

Create `src/lottie/cli/tests/test_status.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.cli import app

runner = CliRunner()


def test_status_lists_units(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    monkeypatch.chdir(tmp_path / "demo")
    assert runner.invoke(app, ["create", "agent", "researcher"]).exit_code == 0
    assert runner.invoke(app, ["create", "skill", "cleaner"]).exit_code == 0

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.output
    assert "demo" in result.output
    assert "researcher" in result.output
    assert "cleaner" in result.output


def test_status_empty_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    monkeypatch.chdir(tmp_path / "demo")
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.output
    assert "No agents" in result.output


def test_status_outside_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["status"])
    assert result.exit_code != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/cli/tests/test_status.py -v`
Expected: FAIL — `status` is not a known command (exit code 2).

- [ ] **Step 3: Implement**

Create `src/lottie/cli/status.py`:

```python
"""`lottie status` — show project config, registered units, and knowledge size."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from lottie.project.config import find_project_root, load_lottie_config
from lottie.project.discovery import UnitInfo, discover_agents, discover_skills


def status() -> None:
    """Show registered agents, skills, knowledge size, and provider config."""
    root = find_project_root()
    cfg = load_lottie_config(root)
    console = Console()
    console.print(f"[bold]{cfg.project}[/bold]")
    console.print(
        f"providers: default={cfg.providers.default} "
        f"fallback={cfg.providers.fallback or '—'}"
    )
    console.print(f"policies: {', '.join(cfg.policies) or '—'}")
    _print_units(console, "Agents", discover_agents(root))
    _print_units(console, "Skills", discover_skills(root))
    _print_knowledge(console, root)


def _print_units(console: Console, title: str, units: list[UnitInfo]) -> None:
    if not units:
        console.print(f"\n[bold]{title}[/bold]: _No {title.lower()} yet._")
        return
    table = Table(title=title)
    table.add_column("name")
    table.add_column("provider")
    for unit in units:
        table.add_row(unit.name, unit.provider or "—")
    console.print(table)


def _print_knowledge(console: Console, root: Path) -> None:
    kdir = root / "knowledge"
    if not kdir.is_dir():
        return
    console.print("\n[bold]Knowledge[/bold]")
    for layer in sorted(p for p in kdir.iterdir() if p.is_dir()):
        count = sum(1 for f in layer.iterdir() if f.is_file() and f.name != ".gitkeep")
        console.print(f"  {layer.name}: {count}")
```

Modify `src/lottie/cli/app.py`. Add the import beside the existing command imports:

```python
from lottie.cli.status import status
```

And register it after the `app.add_typer(create_app, name="create")` line:

```python
app.command("status")(status)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/cli/tests/test_status.py -v`
Expected: PASS (3 tests).
Then: `uv run mypy --strict src/lottie/cli` and `uv run ruff check src/lottie/cli` — clean.

- [ ] **Step 5: Commit**

```bash
git add src/lottie/cli/status.py src/lottie/cli/app.py src/lottie/cli/tests/test_status.py
git commit -m "feat(cli): add lottie status command"
```

---

## Task 6: `lottie doctor`

**Files:**
- Create: `src/lottie/cli/doctor.py`
- Modify: `src/lottie/cli/app.py`
- Test: `src/lottie/cli/tests/test_doctor.py`

- [ ] **Step 1: Write the failing test**

Create `src/lottie/cli/tests/test_doctor.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.cli import app

runner = CliRunner()


def test_doctor_passes_with_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    monkeypatch.chdir(tmp_path / "demo")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_doctor_fails_on_missing_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    monkeypatch.chdir(tmp_path / "demo")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "MISSING" in result.output


def test_doctor_outside_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    result = runner.invoke(app, ["doctor"])
    # Outside a project: not fatal, just reported.
    assert result.exit_code == 0, result.output
    assert "not in a Lottie project" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/cli/tests/test_doctor.py -v`
Expected: FAIL — `doctor` is not a known command (exit code 2).

- [ ] **Step 3: Implement**

Create `src/lottie/cli/doctor.py`:

```python
"""`lottie doctor` — environment health checks (no live network)."""

from __future__ import annotations

import importlib.util
import os
import sys

import typer
from rich.console import Console

from lottie.project.config import find_project_root, load_lottie_config

# Provider prefix -> required env var. None means no key needed (e.g. local).
_PROVIDER_ENV: dict[str, str | None] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "ollama": None,
}

# (label, ok, detail)
Check = tuple[str, bool, str]


def doctor() -> None:
    """Check environment health — Python, deps, project, API keys."""
    checks: list[Check] = []
    warnings: list[str] = []

    py_ok = sys.version_info >= (3, 12)
    checks.append(("Python >= 3.12", py_ok, f"{sys.version_info.major}.{sys.version_info.minor}"))

    for dep in ("litellm", "jinja2", "pydantic", "yaml"):
        ok = importlib.util.find_spec(dep) is not None
        checks.append((f"dep: {dep}", ok, "installed" if ok else "MISSING"))

    project_checks, project_warnings = _project_checks()
    checks.extend(project_checks)
    warnings.extend(project_warnings)

    _render(Console(), checks, warnings)
    if any(not ok for _, ok, _ in checks):
        raise typer.Exit(1)


def _project_checks() -> tuple[list[Check], list[str]]:
    checks: list[Check] = []
    warnings: list[str] = []
    try:
        root = find_project_root()
    except typer.BadParameter:
        warnings.append("not in a Lottie project — skipping project checks")
        return checks, warnings
    checks.append(("Lottie project", True, str(root)))
    cfg = load_lottie_config(root)
    models = [cfg.providers.default]
    if cfg.providers.fallback:
        models.append(cfg.providers.fallback)
    for model in models:
        prefix = model.split("/")[0]
        if prefix not in _PROVIDER_ENV:
            warnings.append(f"unknown provider '{prefix}' — set its API key manually")
            continue
        env = _PROVIDER_ENV[prefix]
        if env is None:
            checks.append((f"key: {prefix}", True, "no key needed"))
        else:
            present = bool(os.environ.get(env))
            checks.append((f"key: {env}", present, "set" if present else "MISSING"))
    return checks, warnings


def _render(console: Console, checks: list[Check], warnings: list[str]) -> None:
    for label, ok, detail in checks:
        mark = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
        console.print(f"{mark}  {label}  ({detail})")
    for warning in warnings:
        console.print(f"[yellow]WARN[/yellow]  {warning}")
```

Modify `src/lottie/cli/app.py`. Add the import beside the others:

```python
from lottie.cli.doctor import doctor
```

And register it:

```python
app.command("doctor")(doctor)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/cli/tests/test_doctor.py -v`
Expected: PASS (3 tests).
Then: `uv run mypy --strict src/lottie/cli` and `uv run ruff check src/lottie/cli` — clean.

- [ ] **Step 5: Commit**

```bash
git add src/lottie/cli/doctor.py src/lottie/cli/app.py src/lottie/cli/tests/test_doctor.py
git commit -m "feat(cli): add lottie doctor command"
```

---

## Task 7: `lottie run`

**Files:**
- Create: `src/lottie/cli/run.py`
- Modify: `src/lottie/cli/app.py`
- Test: `src/lottie/cli/tests/test_run.py`

- [ ] **Step 1: Write the failing test**

Create `src/lottie/cli/tests/test_run.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from lottie.cli import app

runner = CliRunner()


def _scaffold_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "echo"]).exit_code == 0
    return demo


def _fake_completion_factory(
    captured: dict[str, object],
) -> Callable[..., SimpleNamespace]:
    def fake_completion(model: str, messages: object, **kwargs: object) -> SimpleNamespace:
        captured["model"] = model
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hello world"))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
        )

    return fake_completion


def test_run_executes_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _scaffold_agent(tmp_path, monkeypatch)
    captured: dict[str, object] = {}
    monkeypatch.setattr("litellm.completion", _fake_completion_factory(captured))

    result = runner.invoke(app, ["run", "echo", "--input", '{"query": "hi"}'])
    assert result.exit_code == 0, result.output
    assert "hello world" in result.output
    assert captured["model"] == "anthropic/claude-sonnet-4-6"


def test_run_provider_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _scaffold_agent(tmp_path, monkeypatch)
    captured: dict[str, object] = {}
    monkeypatch.setattr("litellm.completion", _fake_completion_factory(captured))

    result = runner.invoke(
        app, ["run", "echo", "--input", '{"query": "hi"}', "--provider", "openai/gpt-4o"]
    )
    assert result.exit_code == 0, result.output
    assert captured["model"] == "openai/gpt-4o"


def test_run_unknown_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _scaffold_agent(tmp_path, monkeypatch)
    result = runner.invoke(app, ["run", "nope", "--input", "{}"])
    assert result.exit_code != 0


def test_run_malformed_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _scaffold_agent(tmp_path, monkeypatch)
    result = runner.invoke(app, ["run", "echo", "--input", "{not json"])
    assert result.exit_code != 0


def test_run_missing_required_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _scaffold_agent(tmp_path, monkeypatch)
    result = runner.invoke(app, ["run", "echo"])
    assert result.exit_code != 0
    assert "query" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/cli/tests/test_run.py -v`
Expected: FAIL — `run` is not a known command (exit code 2).

- [ ] **Step 3: Implement**

Create `src/lottie/cli/run.py`:

```python
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
    agent = agent_cls(llm=llm)
    result = agent.run(data)
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
```

Modify `src/lottie/cli/app.py`. Add the import:

```python
from lottie.cli.run import run
```

And register it:

```python
app.command("run")(run)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/cli/tests/test_run.py -v`
Expected: PASS (5 tests).
Then: `uv run pytest -q`, `uv run mypy --strict src/lottie`, `uv run ruff check src/lottie` — all clean.

- [ ] **Step 5: Commit**

```bash
git add src/lottie/cli/run.py src/lottie/cli/app.py src/lottie/cli/tests/test_run.py
git commit -m "feat(cli): add lottie run command"
```

---

## Task 8: Verification gates + manual smoke

**Files:** none (verification only)

- [ ] **Step 1: Full test suite**

Run: `uv run pytest -q`
Expected: PASS — all prior tests plus the new project + cli tests green.

- [ ] **Step 2: Type check**

Run: `uv run mypy --strict src/lottie`
Expected: `Success: no issues found`.

- [ ] **Step 3: Lint**

Run: `uv run ruff check src/lottie`
Expected: `All checks passed!`

- [ ] **Step 4: Manual smoke — status + doctor (no network)**

Run:
```bash
REPO=/Users/cdiaz19/Documents/trae_projects/lottie-orchestrator
TMP=$(mktemp -d); cd "$TMP"
uv run --project "$REPO" lottie init demo >/dev/null 2>&1; cd demo
uv run --project "$REPO" lottie create agent researcher >/dev/null 2>&1
uv run --project "$REPO" lottie create skill cleaner >/dev/null 2>&1
echo "--- status ---"; uv run --project "$REPO" lottie status
echo "--- doctor ---"; uv run --project "$REPO" lottie doctor; echo "exit=$?"
cd /; rm -rf "$TMP"
```
Expected: `status` lists `researcher` + `cleaner` + provider config + knowledge counts; `doctor`
prints OK/FAIL/WARN lines (FAIL on any unset API key) and a matching exit code. Clean up the temp
dir (the `rm -rf` above).

- [ ] **Step 5: Manual smoke — run (real LLM, optional)**

Only if an API key is set. Run in the same kind of temp project:
```bash
uv run --project "$REPO" lottie run researcher --input '{"query": "say hi in 3 words"}'
```
Expected: prints the agent Output as JSON (`{"result": "..."}`). If no key is set, this errors at
the provider call — expected; the monkeypatched test covers the wiring.

- [ ] **Step 6: No commit**

Verification only — nothing to commit unless a gate forced a fix (then commit with the matching
`fix(...)` message).
```
