# `lottie init` CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `lottie init <name>` — the first CLI command — which scaffolds a new Lottie project skeleton (dir tree, `lottie.yaml`, `LOTTIE.md`, policies, `.gitignore`), and wire the `lottie` console entry point.

**Architecture:** A Typer app at `src/lottie/cli/app.py` is exposed as the `lottie` console script. Subcommands live in their own modules and register on the shared `app`. The `init` command (`cli/init.py`) resolves a target dir, guards against clobbering, then writes a fixed tree from string templates (`cli/templates.py`). Skeleton only — no agent is generated yet.

**Tech Stack:** Python 3.12, Typer (CLI), `typer.testing.CliRunner` + pytest (tests, no LLM), uv (runner). PyYAML is **not** a dependency — generated YAML files are written as plain strings, and tests assert on their contents with substring/line checks (no parsing).

---

## File structure

| File | Responsibility |
|---|---|
| `pyproject.toml` (modify) | add `[project.scripts] lottie = "lottie.cli:app"` |
| `src/lottie/cli/__init__.py` (modify) | re-export `app` so `lottie.cli:app` resolves |
| `src/lottie/cli/app.py` (create) | Typer `app`; registers subcommands |
| `src/lottie/cli/init.py` (create) | `init` command: resolve target → guard → scaffold |
| `src/lottie/cli/templates.py` (create) | scaffold file contents as string constants |
| `src/lottie/cli/tests/__init__.py` (create) | makes tests a package |
| `src/lottie/cli/tests/test_init.py` (create) | CliRunner tests, no LLM |

---

## Task 1: Typer app + console entry point

**Files:**
- Create: `src/lottie/cli/app.py`
- Modify: `src/lottie/cli/__init__.py`
- Modify: `pyproject.toml`
- Create: `src/lottie/cli/tests/__init__.py`
- Test: `src/lottie/cli/tests/test_init.py`

- [ ] **Step 1: Write the failing test**

Create `src/lottie/cli/tests/__init__.py` (empty file).

Create `src/lottie/cli/tests/test_init.py`:

```python
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.cli import app

runner = CliRunner()


def test_app_exposes_init_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "init" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/cli/tests/test_init.py::test_app_exposes_init_command -v`
Expected: FAIL — `ImportError: cannot import name 'app' from 'lottie.cli'`

- [ ] **Step 3: Write minimal implementation**

Create `src/lottie/cli/app.py`:

```python
"""Lottie CLI — Typer application.

Single `app` instance exposed as the `lottie` console script. Subcommands
live in sibling modules and register here.
"""

from __future__ import annotations

import typer

from lottie.cli.init import init

app = typer.Typer(
    help="Lottie AI Orchestrator",
    no_args_is_help=True,
    add_completion=False,
)
app.command()(init)
```

Replace `src/lottie/cli/__init__.py` (currently empty) with:

```python
from lottie.cli.app import app

__all__ = ["app"]
```

Create a placeholder `src/lottie/cli/init.py` so the import in `app.py` resolves
(real body lands in Task 2):

```python
"""`lottie init` — scaffold a new Lottie project skeleton."""

from __future__ import annotations


def init(name: str) -> None:
    """Scaffold a new Lottie project (implemented in Task 2)."""
    raise NotImplementedError
```

Add to `pyproject.toml` immediately after the `[project]` table's closing field
(after the `dependencies = [...]` block, before `[dependency-groups]`):

```toml
[project.scripts]
lottie = "lottie.cli:app"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/cli/tests/test_init.py::test_app_exposes_init_command -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lottie/cli pyproject.toml
git commit -m "feat(cli): wire lottie Typer app and console entry point"
```

---

## Task 2: `init <name>` scaffolds a project skeleton (default subdir)

**Files:**
- Create: `src/lottie/cli/templates.py`
- Modify: `src/lottie/cli/init.py`
- Test: `src/lottie/cli/tests/test_init.py`

- [ ] **Step 1: Write the failing test**

Append to `src/lottie/cli/tests/test_init.py`:

```python
def test_init_creates_project_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "demo"])
    assert result.exit_code == 0, result.output

    root = tmp_path / "demo"
    expected = [
        "lottie.yaml",
        "LOTTIE.md",
        ".gitignore",
        "agents/__init__.py",
        "skills/__init__.py",
        "policies/base.yaml",
        "knowledge/global/.gitkeep",
        "knowledge/platform/.gitkeep",
        "knowledge/project/.gitkeep",
        "knowledge/memory/.gitkeep",
        "knowledge/draft/.gitkeep",
    ]
    for rel in expected:
        assert (root / rel).is_file(), f"missing {rel}"


def test_init_lottie_yaml_records_name_and_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "demo"])
    text = (tmp_path / "demo" / "lottie.yaml").read_text()
    assert "project: demo" in text
    assert "default: anthropic/claude-sonnet-4-6" in text
    assert "fallback: openai/gpt-4o" in text


def test_init_gitignore_has_runtime_and_private_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "demo"])
    text = (tmp_path / "demo" / ".gitignore").read_text()
    assert ".lottie/" in text
    assert ".private-journey/" in text


def test_init_base_policy_has_rule_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "demo"])
    text = (tmp_path / "demo" / "policies" / "base.yaml").read_text()
    for key in ("allow:", "deny:", "escalate:"):
        assert key in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/cli/tests/test_init.py -v -k "tree or providers or gitignore or policy"`
Expected: FAIL — `init` raises `NotImplementedError` (exit code ≠ 0).

- [ ] **Step 3: Write minimal implementation**

Create `src/lottie/cli/templates.py`:

```python
"""Scaffold file contents for `lottie init`.

Static string constants; the two that need the project name use
`str.format(name=...)`. Kept separate from command logic so the future
`lottie create` generator can reuse them.
"""

from __future__ import annotations

KNOWLEDGE_LAYERS = ["global", "platform", "project", "memory", "draft"]

LOTTIE_YAML = """\
project: {name}
providers:
  default: anthropic/claude-sonnet-4-6
  fallback: openai/gpt-4o
policies:
  - base
registry:
  agents: agents/
  skills: skills/
"""

LOTTIE_MD = """\
# {name}

> A Lottie project. This file is read automatically by all AI tools.

## Agents
_None yet — scaffold one with `lottie create agent <name>`._

## Skills
_None yet — scaffold one with `lottie create skill <name>`._
"""

GITIGNORE = """\
# Lottie runtime
.lottie/

# Private AI context
.private-journey/

# Personal Claude Code settings
.claude/settings.local.json

# Python
__pycache__/
.venv/
"""

POLICY_BASE = """\
# Base governance policy. Rules: allow / deny / escalate.
name: base
allow: []
deny: []
escalate: []
"""

AGENTS_INIT = '"""Auto-discovers and registers all agents in this project."""\n'

SKILLS_INIT = '"""Auto-discovers and registers all skills in this project."""\n'
```

Replace `src/lottie/cli/init.py` with:

```python
"""`lottie init` — scaffold a new Lottie project skeleton."""

from __future__ import annotations

from pathlib import Path

import typer

from lottie.cli import templates


def init(name: str) -> None:
    """Scaffold a new Lottie project skeleton in ./<name>/."""
    target = Path.cwd() / name
    _scaffold(target, name)
    typer.echo(f"Created Lottie project at {target}")
    typer.echo("Next: cd into it and run `lottie create agent <name>`.")


def _scaffold(target: Path, name: str) -> None:
    target.mkdir(parents=True, exist_ok=True)
    (target / "lottie.yaml").write_text(templates.LOTTIE_YAML.format(name=name))
    (target / "LOTTIE.md").write_text(templates.LOTTIE_MD.format(name=name))
    (target / ".gitignore").write_text(templates.GITIGNORE)

    (target / "agents").mkdir(exist_ok=True)
    (target / "agents" / "__init__.py").write_text(templates.AGENTS_INIT)
    (target / "skills").mkdir(exist_ok=True)
    (target / "skills" / "__init__.py").write_text(templates.SKILLS_INIT)

    (target / "policies").mkdir(exist_ok=True)
    (target / "policies" / "base.yaml").write_text(templates.POLICY_BASE)

    for layer in templates.KNOWLEDGE_LAYERS:
        layer_dir = target / "knowledge" / layer
        layer_dir.mkdir(parents=True, exist_ok=True)
        (layer_dir / ".gitkeep").write_text("")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/cli/tests/test_init.py -v`
Expected: PASS (all tests so far green).

- [ ] **Step 5: Commit**

```bash
git add src/lottie/cli
git commit -m "feat(cli): scaffold project skeleton in lottie init"
```

---

## Task 3: `--here` scaffolds into the current directory

**Files:**
- Modify: `src/lottie/cli/init.py`
- Test: `src/lottie/cli/tests/test_init.py`

- [ ] **Step 1: Write the failing test**

Append to `src/lottie/cli/tests/test_init.py`:

```python
def test_init_here_scaffolds_into_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "demo", "--here"])
    assert result.exit_code == 0, result.output

    # No nested ./demo/ subdir — files land directly in cwd.
    assert not (tmp_path / "demo").exists()
    assert (tmp_path / "lottie.yaml").is_file()
    assert "project: demo" in (tmp_path / "lottie.yaml").read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/cli/tests/test_init.py::test_init_here_scaffolds_into_cwd -v`
Expected: FAIL — `--here` is not a known option (exit code 2) / or files land in `./demo/`.

- [ ] **Step 3: Write minimal implementation**

In `src/lottie/cli/init.py`, replace the `init` function signature and target
resolution to add the `--here` option. Replace the whole `init` function with:

```python
from typing import Annotated  # add to imports at top of file


def init(
    name: str,
    here: Annotated[
        bool, typer.Option("--here", help="Scaffold into the current directory.")
    ] = False,
) -> None:
    """Scaffold a new Lottie project skeleton."""
    target = Path.cwd() if here else Path.cwd() / name
    _scaffold(target, name)
    typer.echo(f"Created Lottie project at {target}")
    typer.echo("Next: cd into it and run `lottie create agent <name>`.")
```

Place the `from typing import Annotated` import with the other top-of-file imports
(after `from __future__ import annotations`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/cli/tests/test_init.py -v`
Expected: PASS (all tests green).

- [ ] **Step 5: Commit**

```bash
git add src/lottie/cli
git commit -m "feat(cli): add --here flag to lottie init"
```

---

## Task 4: Guards — refuse clobbering an existing project

**Files:**
- Modify: `src/lottie/cli/init.py`
- Test: `src/lottie/cli/tests/test_init.py`

- [ ] **Step 1: Write the failing test**

Append to `src/lottie/cli/tests/test_init.py`:

```python
def test_init_refuses_non_empty_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    existing = tmp_path / "demo"
    existing.mkdir()
    (existing / "keep.txt").write_text("important")

    result = runner.invoke(app, ["init", "demo"])

    assert result.exit_code != 0
    # Nothing clobbered, no scaffold written.
    assert (existing / "keep.txt").read_text() == "important"
    assert not (existing / "lottie.yaml").exists()


def test_init_here_refuses_existing_lottie_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "lottie.yaml").write_text("project: already-here\n")

    result = runner.invoke(app, ["init", "demo", "--here"])

    assert result.exit_code != 0
    # Existing config untouched.
    assert (tmp_path / "lottie.yaml").read_text() == "project: already-here\n"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/cli/tests/test_init.py -v -k "refuses"`
Expected: FAIL — no guard yet; `init` overwrites / scaffolds anyway (exit code 0, assertions fail).

- [ ] **Step 3: Write minimal implementation**

In `src/lottie/cli/init.py`, add a `_guard` call before `_scaffold` in `init`, and
define the `_guard` helper. The `init` body becomes:

```python
    target = Path.cwd() if here else Path.cwd() / name
    _guard(target, here)
    _scaffold(target, name)
```

Add this helper (place it above `_scaffold`):

```python
def _guard(target: Path, here: bool) -> None:
    """Refuse to scaffold over an existing project — fail before any write."""
    if here:
        if (target / "lottie.yaml").exists():
            raise typer.BadParameter(
                f"{target} already contains a lottie.yaml — refusing to overwrite."
            )
        return
    if target.exists() and any(target.iterdir()):
        raise typer.BadParameter(
            f"{target} already exists and is not empty — choose another name."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/cli/tests/test_init.py -v`
Expected: PASS (all tests green).

- [ ] **Step 5: Commit**

```bash
git add src/lottie/cli
git commit -m "feat(cli): guard lottie init against clobbering existing projects"
```

---

## Task 5: Verification gates + manual smoke

**Files:** none (verification only)

- [ ] **Step 1: Full test suite**

Run: `uv run pytest -q`
Expected: PASS — all tests (prior 37 + 7 new cli tests) green.

- [ ] **Step 2: Type check**

Run: `uv run mypy --strict src/lottie/cli`
Expected: `Success: no issues found`.

If mypy flags the Typer option default, confirm the `Annotated[bool, typer.Option(...)] = False`
form is used (not the bare `typer.Option(False, ...)` positional-default form).

- [ ] **Step 3: Lint**

Run: `uv run ruff check src/lottie/cli`
Expected: `All checks passed!`

- [ ] **Step 4: Manual smoke**

Run:
```bash
cd "$(mktemp -d)" && uv run --project /Users/cdiaz19/Documents/trae_projects/lottie-orchestrator lottie init demo && find demo -type f | sort
```
Expected: prints the created tree — `demo/lottie.yaml`, `demo/LOTTIE.md`, `demo/.gitignore`,
`demo/agents/__init__.py`, `demo/skills/__init__.py`, `demo/policies/base.yaml`, and the five
`demo/knowledge/<layer>/.gitkeep` files.

- [ ] **Step 5: No commit**

Verification only — nothing to commit unless a gate forced a fix (then commit that fix with a
`fix(cli):` message).
