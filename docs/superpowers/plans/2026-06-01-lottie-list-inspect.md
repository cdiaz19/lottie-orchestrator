# `lottie list` / `lottie inspect` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `lottie list agents|skills` and `lottie inspect agent|skill <name>` registry-query commands.

**Architecture:** A new `lottie/project/discovery.py` schema-loader pair (`load_schema_models`, `load_system_prompt`) generalizes the existing agent-only `load_input_model`. A new `lottie/cli/registry.py` defines two Typer sub-apps (`list`, `inspect`) wired into `lottie/cli/app.py`. `list` is import-free for agents and per-skill-guarded for skills; `inspect` imports the target unit's code (like `run`).

**Tech Stack:** Python 3.12, Pydantic v2, Typer, Rich, pytest, `typer.testing.CliRunner`. Run all tools from the project dir (`/Users/cdiaz19/Documents/trae_projects/lottie-orchestrator`).

**Reference spec:** `docs/superpowers/specs/2026-06-01-lottie-list-inspect-design.md`

---

## File Structure

- `src/lottie/project/discovery.py` — **modify**: add `_find_model`, `load_schema_models`, `load_system_prompt`; add `kind` param to `_import_unit_module`; reduce `load_input_model` to a wrapper.
- `src/lottie/project/tests/test_discovery.py` — **modify**: add loader tests.
- `src/lottie/cli/registry.py` — **create**: `list_app` + `inspect_app`.
- `src/lottie/cli/app.py` — **modify**: register the two sub-apps.
- `src/lottie/cli/tests/test_registry.py` — **create**: command tests.

Scaffold facts (from `lottie create`): agent `researcher` → class `ResearcherAgent`, schema classes `ResearcherAgentInput` (`query: str`) / `ResearcherAgentOutput` (`result: str`), `prompts.py` exposing `SYSTEM_PROMPT = "You are ResearcherAgent, a Lottie agent. …"`, provider `anthropic/claude-sonnet-4-6`. Skill `cleaner` → class `CleanerSkill`, `CleanerSkillInput` (`text: str`) / `CleanerSkillOutput` (`result: str`), has `SKILL.md`, no provider/prompt.

---

## Task 1: Generalize discovery schema loaders

**Files:**
- Modify: `src/lottie/project/discovery.py`
- Test: `src/lottie/project/tests/test_discovery.py`

- [ ] **Step 1: Write the failing tests**

Add to `src/lottie/project/tests/test_discovery.py`. First extend the import block at the top (currently imports `discover_agents, discover_skills, load_agent_class, load_input_model, required_fields`) to also import the new functions:

```python
from lottie.project.discovery import (
    discover_agents,
    discover_skills,
    load_agent_class,
    load_input_model,
    load_schema_models,
    load_system_prompt,
    required_fields,
)
```

Then append these tests:

```python
def test_load_schema_models_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    in_model, out_model = load_schema_models(demo, "agent", "researcher")
    assert in_model.__name__ == "ResearcherAgentInput"
    assert out_model.__name__ == "ResearcherAgentOutput"
    assert "query" in in_model.model_fields


def test_load_schema_models_skill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    in_model, out_model = load_schema_models(demo, "skill", "cleaner")
    assert in_model.__name__ == "CleanerSkillInput"
    assert out_model.__name__ == "CleanerSkillOutput"
    assert "text" in in_model.model_fields


def test_load_schema_models_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    with pytest.raises(typer.BadParameter):
        load_schema_models(demo, "skill", "nope")


def test_load_system_prompt_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    prompt = load_system_prompt(demo, "researcher")
    assert prompt is not None
    assert "ResearcherAgent" in prompt


def test_load_system_prompt_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    (demo / "agents" / "researcher" / "prompts.py").unlink()
    assert load_system_prompt(demo, "researcher") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/lottie/project/tests/test_discovery.py -k "schema_models or system_prompt" -v`
Expected: FAIL — `ImportError: cannot import name 'load_schema_models'`.

- [ ] **Step 3: Refactor `discovery.py`**

In `src/lottie/project/discovery.py`:

(a) Add a `kind` parameter to `_import_unit_module` so error messages name the unit type. Change its signature and the two error strings:

```python
def _import_unit_module(
    root: Path, dotted: str, name: str, kind: str = "agent"
) -> ModuleType:
```

and inside the `except` handlers replace the two messages:

```python
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(f"{kind} '{name}' not found") from exc
    except ImportError as exc:
        raise typer.BadParameter(f"{kind} '{name}' failed to import: {exc}") from exc
```

(The docstring and caching logic above are unchanged.)

(b) Replace the whole `load_input_model` function (and `required_fields` stays as-is below it) with this block:

```python
def _find_model(
    module: ModuleType, suffix: Literal["Input", "Output"], name: str
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
        f"{name}/schema.py must define an `{suffix}(BaseModel)` class"
        f" or exactly one `<Name>{suffix}(BaseModel)` class"
        f" — found {len(candidates)}"
    )


def load_schema_models(
    root: Path, kind: Literal["agent", "skill"], name: str
) -> tuple[type[BaseModel], type[BaseModel]]:
    """Import {kind}s/<name>/schema.py and return its (Input, Output) models."""
    module = _import_unit_module(root, f"{kind}s.{name}.schema", name, kind)
    return _find_model(module, "Input", name), _find_model(module, "Output", name)


def load_input_model(root: Path, name: str) -> type[BaseModel]:
    """Import agents/<name>/schema.py and return its Input model (used by `run`)."""
    return load_schema_models(root, "agent", name)[0]


def load_system_prompt(root: Path, name: str) -> str | None:
    """Return SYSTEM_PROMPT from agents/<name>/prompts.py, or None if absent.

    A missing prompts.py (or missing/ non-str SYSTEM_PROMPT) yields None so
    `inspect` can render `—` rather than fail.
    """
    try:
        module = _import_unit_module(root, f"agents.{name}.prompts", name)
    except typer.BadParameter:
        return None
    prompt = getattr(module, "SYSTEM_PROMPT", None)
    return prompt if isinstance(prompt, str) else None
```

`Literal` is already imported (`from typing import Literal`). `BaseModel`, `ModuleType`, `typer`, `Path` are already imported.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/lottie/project/tests/test_discovery.py -v`
Expected: PASS — new tests green AND the existing `test_load_input_model_*`, `test_load_agent_class_*` still pass (wrapper preserves behavior).

- [ ] **Step 5: Commit**

```bash
git add src/lottie/project/discovery.py src/lottie/project/tests/test_discovery.py
git commit -m "feat(discovery): add load_schema_models and load_system_prompt loaders"
```

---

## Task 2: `lottie list agents` and `lottie list skills`

**Files:**
- Create: `src/lottie/cli/registry.py`
- Modify: `src/lottie/cli/app.py`
- Test: `src/lottie/cli/tests/test_registry.py`

- [ ] **Step 1: Write the failing tests**

Create `src/lottie/cli/tests/test_registry.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.cli import app

runner = CliRunner()


def _scaffold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "researcher"]).exit_code == 0
    assert runner.invoke(app, ["create", "skill", "cleaner"]).exit_code == 0
    return demo


def test_list_agents_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    monkeypatch.chdir(tmp_path / "demo")
    result = runner.invoke(app, ["list", "agents"])
    assert result.exit_code == 0, result.output
    assert "No agents" in result.output


def test_list_agents_populated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _scaffold(tmp_path, monkeypatch)
    result = runner.invoke(app, ["list", "agents"])
    assert result.exit_code == 0, result.output
    assert "researcher" in result.output
    assert "anthropic" in result.output


def test_list_skills_populated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _scaffold(tmp_path, monkeypatch)
    result = runner.invoke(app, ["list", "skills"])
    assert result.exit_code == 0, result.output
    assert "cleaner" in result.output
    assert "CleanerSkillInput" in result.output
    assert "CleanerSkillOutput" in result.output


def test_list_skills_broken_schema_degrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    (demo / "skills" / "cleaner" / "schema.py").write_text(
        "raise RuntimeError('boom')\n", encoding="utf-8"
    )
    result = runner.invoke(app, ["list", "skills"])
    assert result.exit_code == 0, result.output
    assert "cleaner" in result.output
    assert "—" in result.output


def test_list_outside_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["list", "agents"]).exit_code != 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/lottie/cli/tests/test_registry.py -v`
Expected: FAIL — `list` is not a registered command (non-zero exit / "No such command").

- [ ] **Step 3: Create `registry.py` (list portion) and wire it**

Create `src/lottie/cli/registry.py`:

```python
"""`lottie list` and `lottie inspect` — registry query commands."""

from __future__ import annotations

import typer
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lottie.project.config import find_project_root, load_agent_config
from lottie.project.discovery import (
    discover_agents,
    discover_skills,
    load_schema_models,
    load_system_prompt,
)

list_app = typer.Typer(help="List registered agents or skills.", no_args_is_help=True)
inspect_app = typer.Typer(help="Inspect an agent or skill.", no_args_is_help=True)


@list_app.command("agents")
def list_agents() -> None:
    """List registered agents with their provider."""
    root = find_project_root()
    units = discover_agents(root)
    console = Console()
    if not units:
        console.print("_No agents yet._")
        return
    table = Table(title="Agents")
    table.add_column("name")
    table.add_column("provider")
    for unit in units:
        table.add_row(unit.name, unit.provider or "—")
    console.print(table)


@list_app.command("skills")
def list_skills() -> None:
    """List registered skills with their input/output types."""
    root = find_project_root()
    units = discover_skills(root)
    console = Console()
    if not units:
        console.print("_No skills yet._")
        return
    table = Table(title="Skills")
    table.add_column("name")
    table.add_column("input")
    table.add_column("output")
    for unit in units:
        try:
            in_model, out_model = load_schema_models(root, "skill", unit.name)
            in_name, out_name = in_model.__name__, out_model.__name__
        except Exception:  # noqa: BLE001 — one broken skill must not crash the list
            in_name, out_name = "—", "—"
        table.add_row(unit.name, in_name, out_name)
    console.print(table)
```

Then in `src/lottie/cli/app.py` add the import and registration. After the existing `from lottie.cli.create import create_app` import, add:

```python
from lottie.cli.registry import inspect_app, list_app
```

and after the existing `app.add_typer(create_app, name="create")` line add:

```python
app.add_typer(list_app, name="list")
app.add_typer(inspect_app, name="inspect")
```

(`inspect_app` is registered now even though its commands land in Task 3 — keep both `add_typer` lines together. The `inspect_app` Typer already exists in `registry.py` with `no_args_is_help=True`, so `lottie inspect` is valid but lists no subcommands until Task 3.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/lottie/cli/tests/test_registry.py -v`
Expected: PASS — all five `list` tests green.

- [ ] **Step 5: Commit**

```bash
git add src/lottie/cli/registry.py src/lottie/cli/app.py src/lottie/cli/tests/test_registry.py
git commit -m "feat(cli): add lottie list agents/skills"
```

---

## Task 3: `lottie inspect agent` and `lottie inspect skill`

**Files:**
- Modify: `src/lottie/cli/registry.py`
- Test: `src/lottie/cli/tests/test_registry.py`

- [ ] **Step 1: Write the failing tests**

Append to `src/lottie/cli/tests/test_registry.py`:

```python
def test_inspect_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _scaffold(tmp_path, monkeypatch)
    result = runner.invoke(app, ["inspect", "agent", "researcher"])
    assert result.exit_code == 0, result.output
    assert "anthropic" in result.output      # provider from config.yaml
    assert "query" in result.output          # Input field
    assert "ResearcherAgent" in result.output  # from SYSTEM_PROMPT


def test_inspect_skill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _scaffold(tmp_path, monkeypatch)
    result = runner.invoke(app, ["inspect", "skill", "cleaner"])
    assert result.exit_code == 0, result.output
    assert "text" in result.output           # Input field
    assert "result" in result.output         # Output field


def test_inspect_agent_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _scaffold(tmp_path, monkeypatch)
    assert runner.invoke(app, ["inspect", "agent", "nope"]).exit_code != 0


def test_inspect_skill_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _scaffold(tmp_path, monkeypatch)
    assert runner.invoke(app, ["inspect", "skill", "nope"]).exit_code != 0


def test_inspect_outside_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["inspect", "agent", "researcher"]).exit_code != 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/lottie/cli/tests/test_registry.py -k inspect -v`
Expected: FAIL — `inspect agent`/`inspect skill` are not commands (non-zero exit / "No such command").

- [ ] **Step 3: Add the inspect commands to `registry.py`**

Append to `src/lottie/cli/registry.py`:

```python
def _field_lines(model: type[BaseModel]) -> str:
    """One `name: type` line per field, for inspect output."""
    lines: list[str] = []
    for fname, field in model.model_fields.items():
        ann = field.annotation
        type_name = getattr(ann, "__name__", str(ann))
        lines.append(f"  {fname}: {type_name}")
    return "\n".join(lines) or "  (no fields)"


@inspect_app.command("agent")
def inspect_agent(name: str) -> None:
    """Show an agent's config, schema, and system prompt."""
    root = find_project_root()
    if name not in [u.name for u in discover_agents(root)]:
        raise typer.BadParameter(f"agent '{name}' not found")
    cfg = load_agent_config(root / "agents" / name)
    in_model, out_model = load_schema_models(root, "agent", name)
    prompt = load_system_prompt(root, name) or "—"
    body = (
        f"provider: {cfg.provider}\n"
        f"model_params: {cfg.model_params}\n"
        f"capabilities: {', '.join(cfg.capabilities) or '—'}\n"
        f"policies: {', '.join(cfg.policies) or '—'}\n\n"
        f"Input:\n{_field_lines(in_model)}\n"
        f"Output:\n{_field_lines(out_model)}\n\n"
        f"System prompt:\n{prompt}"
    )
    Console().print(Panel(body, title=f"agent: {name}"))


@inspect_app.command("skill")
def inspect_skill(name: str) -> None:
    """Show a skill's schema and SKILL.md presence."""
    root = find_project_root()
    if name not in [u.name for u in discover_skills(root)]:
        raise typer.BadParameter(f"skill '{name}' not found")
    in_model, out_model = load_schema_models(root, "skill", name)
    has_doc = (root / "skills" / name / "SKILL.md").is_file()
    body = (
        f"SKILL.md: {'present' if has_doc else 'missing'}\n\n"
        f"Input:\n{_field_lines(in_model)}\n"
        f"Output:\n{_field_lines(out_model)}"
    )
    Console().print(Panel(body, title=f"skill: {name}"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/lottie/cli/tests/test_registry.py -v`
Expected: PASS — all `list` and `inspect` tests green.

- [ ] **Step 5: Commit**

```bash
git add src/lottie/cli/registry.py src/lottie/cli/tests/test_registry.py
git commit -m "feat(cli): add lottie inspect agent/skill"
```

---

## Task 4: Full gate — suite, mypy, ruff

**Files:** none (verification only).

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest`
Expected: PASS — all tests (116 prior + the new discovery/registry tests), zero failures.

- [ ] **Step 2: Type-check**

Run: `uv run mypy --strict src/lottie`
Expected: `Success: no issues found`. If `_field_lines`'s `getattr(ann, "__name__", str(ann))` trips strict (returns `Any`), annotate the local: `type_name: str = getattr(ann, "__name__", None) or str(ann)`.

- [ ] **Step 3: Lint**

Run: `uv run ruff check src/lottie`
Expected: `All checks passed!`. The single `# noqa: BLE001` on the broad-except in `list_skills` is intentional (robustness); keep it.

- [ ] **Step 4: Commit any gate fixes**

```bash
git add -A
git commit -m "chore: satisfy mypy --strict and ruff for list/inspect"
```

(Skip this commit if Steps 2–3 needed no changes.)

---

## Notes for the implementer

- Run every command from the project dir, not a workspace root (mypy/discovery footgun).
- `discover_agents`/`discover_skills` are import-free; only `load_schema_models`/`load_system_prompt` import user code. `list skills` guards its import per-row so one broken skill degrades to `—` rather than crashing.
- Do NOT add benchmark / last-run / version columns — there is no store for them yet (deliberately out of scope).
- The em-dash placeholder is `—` (U+2014), matching `status.py`.
