# `lottie create agent/skill` Generators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `lottie create agent <name>` and `lottie create skill <name>` — generators that scaffold a complete, working agent or skill module from Jinja2 templates, guard against clobbering, and register the unit in `LOTTIE.md`.

**Architecture:** A new `src/lottie/scaffold/` package holds a typed `TemplateRendererSkill` (a `BaseSkill` wrapping Jinja2), the `.j2` templates, and a `generator.py` orchestrator (validate → resolve project root → guard → render → write → update `LOTTIE.md`). A thin Typer `create` sub-group in `cli/create.py` delegates to the generator. Deterministic only — `--from-desc`/`ScaffolderAgent` is a later work item.

**Tech Stack:** Python 3.12, Jinja2 (new dep), Typer + `typer.testing.CliRunner` + pytest (tests, no real LLM), Pydantic v2, uv (runner), hatchling (build).

**Spec:** `docs/superpowers/specs/2026-05-29-lottie-create-generators-design.md`

---

## File structure

| File | Responsibility |
|---|---|
| `pyproject.toml` (modify) | add `jinja2` dep + wheel `force-include` for templates |
| `src/lottie/scaffold/__init__.py` (create) | package marker |
| `src/lottie/scaffold/schema.py` (create) | `RenderContext`, `RenderInput`, `RenderOutput` |
| `src/lottie/scaffold/renderer.py` (create) | `TemplateRendererSkill(BaseSkill[...])` |
| `src/lottie/scaffold/generator.py` (create) | orchestrator + name/class/guard helpers |
| `src/lottie/scaffold/templates/agent/*.j2` (create) | agent file templates |
| `src/lottie/scaffold/templates/skill/*.j2` (create) | skill file templates |
| `src/lottie/scaffold/tests/__init__.py` (create) | test package marker |
| `src/lottie/scaffold/tests/test_renderer.py` (create) | renderer unit tests |
| `src/lottie/scaffold/tests/test_generator.py` (create) | name/class helper unit tests |
| `src/lottie/cli/create.py` (create) | thin Typer `create` group |
| `src/lottie/cli/app.py` (modify) | register `create_app` |
| `src/lottie/cli/tests/test_create.py` (create) | CliRunner end-to-end + `py_compile` |

---

## Task 1: Add jinja2 + scaffold package skeleton + packaging

**Files:**
- Modify: `pyproject.toml`
- Create: `src/lottie/scaffold/__init__.py`
- Create: `src/lottie/scaffold/tests/__init__.py`

- [ ] **Step 1: Add the dependency**

Run: `uv add jinja2`
Expected: `pyproject.toml` `[project].dependencies` gains a `jinja2>=...` entry; `uv.lock` updates.

- [ ] **Step 2: Add the wheel force-include for templates**

The `.j2` files are package *data*, not modules. `PackageLoader` resolves them at runtime only if
the build ships them. Add this table to `pyproject.toml` (after the existing
`[tool.hatch.build.targets.wheel]` block):

```toml
[tool.hatch.build.targets.wheel.force-include]
"src/lottie/scaffold/templates" = "lottie/scaffold/templates"
```

- [ ] **Step 3: Create the package markers**

Create `src/lottie/scaffold/__init__.py`:

```python
"""Scaffold generation — Jinja2 templates, renderer skill, and the create generator."""
```

Create `src/lottie/scaffold/tests/__init__.py` (empty file).

- [ ] **Step 4: Verify jinja2 imports**

Run: `uv run python -c "import jinja2; print(jinja2.__version__)"`
Expected: prints a version (e.g. `3.1.x`), no error.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/lottie/scaffold
git commit -m "chore: add jinja2 dep and scaffold package skeleton"
```

---

## Task 2: Render contract schema

**Files:**
- Create: `src/lottie/scaffold/schema.py`
- Test: `src/lottie/scaffold/tests/test_renderer.py`

- [ ] **Step 1: Write the failing test**

Create `src/lottie/scaffold/tests/test_renderer.py`:

```python
from __future__ import annotations

from lottie.scaffold.schema import RenderContext, RenderInput, RenderOutput


def test_render_context_defaults_provider() -> None:
    ctx = RenderContext(name="web_search", class_name="WebSearchSkill")
    assert ctx.provider == "anthropic/claude-sonnet-4-6"


def test_render_input_wraps_context() -> None:
    ctx = RenderContext(name="x", class_name="XSkill")
    inp = RenderInput(template="skill/skill.py.j2", context=ctx)
    assert inp.template == "skill/skill.py.j2"
    assert inp.context.class_name == "XSkill"


def test_render_output_holds_content() -> None:
    assert RenderOutput(content="hello").content == "hello"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/scaffold/tests/test_renderer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.scaffold.schema'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/lottie/scaffold/schema.py`:

```python
"""Typed contract for template rendering.

The skill boundary is fully typed (CLAUDE.md rule 2); the dict handed to Jinja is
internal to the renderer.
"""

from __future__ import annotations

from pydantic import BaseModel


class RenderContext(BaseModel):
    """Variables injected into a scaffold template."""

    name: str
    class_name: str
    provider: str = "anthropic/claude-sonnet-4-6"


class RenderInput(BaseModel):
    """Input to TemplateRendererSkill — which template, with what context."""

    template: str
    context: RenderContext


class RenderOutput(BaseModel):
    """Rendered template content."""

    content: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/scaffold/tests/test_renderer.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lottie/scaffold/schema.py src/lottie/scaffold/tests/test_renderer.py
git commit -m "feat(scaffold): add render contract schema"
```

---

## Task 3: Skill templates + TemplateRendererSkill

Write the skill templates first (the renderer test renders a real one), then the skill.

**Files:**
- Create: `src/lottie/scaffold/templates/skill/SKILL.md.j2`
- Create: `src/lottie/scaffold/templates/skill/skill.py.j2`
- Create: `src/lottie/scaffold/templates/skill/schema.py.j2`
- Create: `src/lottie/scaffold/templates/skill/test.py.j2`
- Create: `src/lottie/scaffold/renderer.py`
- Test: `src/lottie/scaffold/tests/test_renderer.py`

- [ ] **Step 1: Write the failing test**

Append to `src/lottie/scaffold/tests/test_renderer.py`:

```python
import pytest
from jinja2 import StrictUndefined, TemplateError

from lottie.scaffold.renderer import TemplateRendererSkill


def test_renders_class_name_into_skill_template() -> None:
    skill = TemplateRendererSkill()
    ctx = RenderContext(name="web_search", class_name="WebSearchSkill")
    out = skill.run(RenderInput(template="skill/skill.py.j2", context=ctx))
    assert "class WebSearchSkill(BaseSkill" in out.content


def test_environment_uses_strict_undefined() -> None:
    skill = TemplateRendererSkill()
    assert skill._env.undefined is StrictUndefined


def test_unknown_template_raises() -> None:
    skill = TemplateRendererSkill()
    ctx = RenderContext(name="x", class_name="XSkill")
    with pytest.raises(TemplateError):
        skill.run(RenderInput(template="nope/missing.j2", context=ctx))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/scaffold/tests/test_renderer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.scaffold.renderer'`.

- [ ] **Step 3: Create the skill templates**

Create `src/lottie/scaffold/templates/skill/schema.py.j2`:

```jinja
"""Typed input/output models for {{ class_name }}."""

from __future__ import annotations

from pydantic import BaseModel


class Input(BaseModel):
    """Input for {{ class_name }}."""

    text: str


class Output(BaseModel):
    """Output from {{ class_name }}."""

    result: str
```

Create `src/lottie/scaffold/templates/skill/skill.py.j2`:

```jinja
"""{{ class_name }} — generated by `lottie create skill {{ name }}`."""

from __future__ import annotations

from lottie.core import BaseSkill

from .schema import Input, Output


class {{ class_name }}(BaseSkill[Input, Output]):
    """Deterministic skill. Replace `_execute` with real logic."""

    def _execute(self, data: Input) -> Output:
        return Output(result=data.text)
```

Create `src/lottie/scaffold/templates/skill/test.py.j2`:

```jinja
"""Unit tests for {{ class_name }} (deterministic — no LLM)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from skills.{{ name }}.schema import Input
from skills.{{ name }}.skill import {{ class_name }}


def test_{{ name }}_happy_path() -> None:
    skill = {{ class_name }}()
    result = skill.run(Input(text="hello"))
    assert result.result == "hello"


def test_{{ name }}_handles_empty_text() -> None:
    skill = {{ class_name }}()
    result = skill.run(Input(text=""))
    assert result.result == ""


def test_{{ name }}_rejects_wrong_type() -> None:
    with pytest.raises(ValidationError):
        Input(text=123)  # type: ignore[arg-type]
```

Create `src/lottie/scaffold/templates/skill/SKILL.md.j2`:

```jinja
# {{ class_name }}

## What it does
One sentence description.

## Input
| Field | Type | Required | Description |
|---|---|---|---|
| text | str | yes | Input text |

## Output
| Field | Type | Description |
|---|---|---|
| result | str | Processed text |

## Side effects
None.

## Examples
### Example 1
Input: `{"text": "hello"}`
Output: `{"result": "hello"}`
```

- [ ] **Step 4: Write the renderer**

Create `src/lottie/scaffold/renderer.py`:

```python
"""TemplateRendererSkill — renders Jinja2 scaffold templates with a typed contract.

Templates are package data under `scaffold/templates/`, loaded via `PackageLoader`
so rendering works in an installed wheel, not just editable dev.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, PackageLoader, StrictUndefined

from lottie.core import BaseSkill
from lottie.scaffold.schema import RenderInput, RenderOutput


class TemplateRendererSkill(BaseSkill[RenderInput, RenderOutput]):
    """Render a named template against a typed context."""

    def __init__(
        self,
        *,
        name: str | None = None,
        enable_benchmarks: bool | None = None,
        benchmarks_root: Path | None = None,
    ) -> None:
        super().__init__(
            name=name,
            enable_benchmarks=enable_benchmarks,
            benchmarks_root=benchmarks_root,
        )
        self._env = Environment(
            loader=PackageLoader("lottie.scaffold", "templates"),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
            autoescape=False,  # rendering Python/Markdown/YAML, never HTML
        )

    def _execute(self, data: RenderInput) -> RenderOutput:
        template = self._env.get_template(data.template)
        return RenderOutput(content=template.render(**data.context.model_dump()))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest src/lottie/scaffold/tests/test_renderer.py -v`
Expected: PASS (6 tests total).

- [ ] **Step 6: Commit**

```bash
git add src/lottie/scaffold/renderer.py src/lottie/scaffold/templates/skill src/lottie/scaffold/tests/test_renderer.py
git commit -m "feat(scaffold): add TemplateRendererSkill and skill templates"
```

---

## Task 4: Agent templates

**Files:**
- Create: `src/lottie/scaffold/templates/agent/AGENT.md.j2`
- Create: `src/lottie/scaffold/templates/agent/agent.py.j2`
- Create: `src/lottie/scaffold/templates/agent/schema.py.j2`
- Create: `src/lottie/scaffold/templates/agent/config.yaml.j2`
- Create: `src/lottie/scaffold/templates/agent/prompts.py.j2`
- Create: `src/lottie/scaffold/templates/agent/test.py.j2`
- Test: `src/lottie/scaffold/tests/test_renderer.py`

- [ ] **Step 1: Write the failing test**

Append to `src/lottie/scaffold/tests/test_renderer.py`:

```python
def test_renders_agent_class_and_provider() -> None:
    skill = TemplateRendererSkill()
    ctx = RenderContext(name="researcher", class_name="ResearcherAgent")
    agent_py = skill.run(RenderInput(template="agent/agent.py.j2", context=ctx))
    assert "class ResearcherAgent(BaseAgent" in agent_py.content
    config = skill.run(RenderInput(template="agent/config.yaml.j2", context=ctx))
    assert "provider: anthropic/claude-sonnet-4-6" in config.content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/scaffold/tests/test_renderer.py::test_renders_agent_class_and_provider -v`
Expected: FAIL — `jinja2.exceptions.TemplateNotFound: agent/agent.py.j2`.

- [ ] **Step 3: Create the agent templates**

Create `src/lottie/scaffold/templates/agent/schema.py.j2`:

```jinja
"""Typed input/output models for {{ class_name }}."""

from __future__ import annotations

from pydantic import BaseModel


class Input(BaseModel):
    """Input for {{ class_name }}."""

    query: str


class Output(BaseModel):
    """Output from {{ class_name }}."""

    result: str
```

Create `src/lottie/scaffold/templates/agent/prompts.py.j2`:

```jinja
"""Prompt templates for {{ class_name }}."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are {{ class_name }}, a Lottie agent.
Answer the user's query concisely and accurately.
"""
```

Create `src/lottie/scaffold/templates/agent/agent.py.j2`:

```jinja
"""{{ class_name }} — generated by `lottie create agent {{ name }}`."""

from __future__ import annotations

from lottie.core import BaseAgent
from lottie.llm import Message

from .prompts import SYSTEM_PROMPT
from .schema import Input, Output


class {{ class_name }}(BaseAgent[Input, Output]):
    """LLM-backed agent. Replace `_execute` with real reasoning."""

    def _execute(self, data: Input) -> Output:
        response = self.complete(
            [
                Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content=data.query),
            ]
        )
        return Output(result=response.content)
```

Create `src/lottie/scaffold/templates/agent/config.yaml.j2`:

```jinja
provider: {{ provider }}
model_params:
  temperature: 0.3
  max_tokens: 2048
capabilities: []
policies:
  - base
memory:
  enabled: false
  namespace: {{ name }}
```

Create `src/lottie/scaffold/templates/agent/test.py.j2`:

```jinja
"""Integration tests for {{ class_name }} (MockLLMProvider — no real LLM)."""

from __future__ import annotations

from lottie.llm import MockLLMProvider

from agents.{{ name }}.agent import {{ class_name }}
from agents.{{ name }}.schema import Input


def test_{{ name }}_returns_llm_content() -> None:
    agent = {{ class_name }}(llm=MockLLMProvider(["hello from mock"]))
    result = agent.run(Input(query="hi"))
    assert result.result == "hello from mock"


def test_{{ name }}_makes_one_llm_call() -> None:
    mock = MockLLMProvider(["ok"])
    agent = {{ class_name }}(llm=mock)
    agent.run(Input(query="hi"))
    assert len(mock.calls) == 1


def test_{{ name }}_handles_empty_query() -> None:
    agent = {{ class_name }}(llm=MockLLMProvider(["fallback"]))
    result = agent.run(Input(query=""))
    assert result.result == "fallback"
```

Create `src/lottie/scaffold/templates/agent/AGENT.md.j2`:

```jinja
# {{ class_name }}

## Role
One sentence describing what this agent does.

## Input
| Field | Type | Description |
|---|---|---|
| query | str | The user's query |

## Output
| Field | Type | Description |
|---|---|---|
| result | str | The agent's response |

## Provider
Default: {{ provider }}

## Tools (Skills used)
_None yet._

## Policies
- base

## Examples
### Example 1
Input: `{"query": "..."}`
Output: `{"result": "..."}`
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/scaffold/tests/test_renderer.py -v`
Expected: PASS (7 tests total).

- [ ] **Step 5: Commit**

```bash
git add src/lottie/scaffold/templates/agent src/lottie/scaffold/tests/test_renderer.py
git commit -m "feat(scaffold): add agent templates"
```

---

## Task 5: Generator helpers — name validation + class derivation

**Files:**
- Create: `src/lottie/scaffold/generator.py`
- Test: `src/lottie/scaffold/tests/test_generator.py`

- [ ] **Step 1: Write the failing test**

Create `src/lottie/scaffold/tests/test_generator.py`:

```python
from __future__ import annotations

from typing import Literal

import pytest
import typer

from lottie.scaffold.generator import _class_name, _validate_name


@pytest.mark.parametrize(
    ("name", "kind", "expected"),
    [
        ("web_search", "skill", "WebSearchSkill"),
        ("researcher", "agent", "ResearcherAgent"),
        ("a_b_c", "skill", "ABCSkill"),
    ],
)
def test_class_name_derivation(
    name: str, kind: Literal["agent", "skill"], expected: str
) -> None:
    assert _class_name(name, kind) == expected


@pytest.mark.parametrize(
    "bad", ["", ".", "..", "a/b", "../x", "Web", "web-search", "1foo"]
)
def test_validate_name_rejects(bad: str) -> None:
    with pytest.raises(typer.BadParameter):
        _validate_name(bad)


def test_validate_name_accepts_snake() -> None:
    _validate_name("web_search")  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/scaffold/tests/test_generator.py -v`
Expected: FAIL — `ImportError: cannot import name '_class_name' from 'lottie.scaffold.generator'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/lottie/scaffold/generator.py`:

```python
"""Scaffold a complete agent or skill module from Jinja2 templates."""

from __future__ import annotations

from typing import Literal

import typer

Kind = Literal["agent", "skill"]


def _validate_name(name: str) -> None:
    """Name must be a single lowercase snake_case Python identifier."""
    if (
        name in ("", ".", "..")
        or name != name.strip()
        or "/" in name
        or "\\" in name
        or not name.isidentifier()
        or not name.islower()
    ):
        raise typer.BadParameter(
            f"{name!r} is not a valid name — use a lowercase snake_case identifier "
            "(e.g. web_search)."
        )


def _class_name(name: str, kind: Kind) -> str:
    """web_search + skill -> WebSearchSkill."""
    pascal = "".join(part.capitalize() for part in name.split("_"))
    return f"{pascal}{kind.capitalize()}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/scaffold/tests/test_generator.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lottie/scaffold/generator.py src/lottie/scaffold/tests/test_generator.py
git commit -m "feat(scaffold): add name validation and class-name derivation"
```

---

## Task 6: Generator core + CLI wiring

**Files:**
- Modify: `src/lottie/scaffold/generator.py`
- Create: `src/lottie/cli/create.py`
- Modify: `src/lottie/cli/app.py`
- Test: `src/lottie/cli/tests/test_create.py`

- [ ] **Step 1: Write the failing test**

Create `src/lottie/cli/tests/test_create.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.cli import app

runner = CliRunner()


def _init_demo(tmp_path: Path) -> Path:
    result = runner.invoke(app, ["init", "demo"])
    assert result.exit_code == 0, result.output
    return tmp_path / "demo"


def test_create_agent_scaffolds_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    demo = _init_demo(tmp_path)
    monkeypatch.chdir(demo)
    result = runner.invoke(app, ["create", "agent", "web_search"])
    assert result.exit_code == 0, result.output

    base = demo / "agents" / "web_search"
    for rel in [
        "AGENT.md",
        "agent.py",
        "schema.py",
        "config.yaml",
        "prompts.py",
        "tests/__init__.py",
        "tests/test_web_search.py",
    ]:
        assert (base / rel).is_file(), f"missing {rel}"
    assert "class WebSearchAgent(BaseAgent" in (base / "agent.py").read_text()


def test_create_skill_scaffolds_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    demo = _init_demo(tmp_path)
    monkeypatch.chdir(demo)
    result = runner.invoke(app, ["create", "skill", "cleaner"])
    assert result.exit_code == 0, result.output

    base = demo / "skills" / "cleaner"
    for rel in ["SKILL.md", "skill.py", "schema.py", "tests/__init__.py", "tests/test_cleaner.py"]:
        assert (base / rel).is_file(), f"missing {rel}"
    assert "class CleanerSkill(BaseSkill" in (base / "skill.py").read_text()


def test_create_refuses_outside_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # no lottie.yaml here
    result = runner.invoke(app, ["create", "agent", "foo"])
    assert result.exit_code != 0
    assert not (tmp_path / "agents").exists()


def test_create_refuses_existing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    demo = _init_demo(tmp_path)
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "dup"]).exit_code == 0
    result = runner.invoke(app, ["create", "agent", "dup"])
    assert result.exit_code != 0
    output = " ".join(result.output.split())
    assert "not empty" in output


@pytest.mark.parametrize("bad", ["Web", "web-search", "a/b", ""])
def test_create_rejects_bad_names(
    bad: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    demo = _init_demo(tmp_path)
    monkeypatch.chdir(demo)
    result = runner.invoke(app, ["create", "agent", bad])
    assert result.exit_code != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/cli/tests/test_create.py -v`
Expected: FAIL — `create` is not a known command (exit code 2).

- [ ] **Step 3: Add the generator orchestrator**

Append to `src/lottie/scaffold/generator.py` (add the new imports at the top of the file,
beside the existing ones):

```python
from pathlib import Path  # add to top-of-file imports

from jinja2 import TemplateError  # add to top-of-file imports

from lottie.scaffold.renderer import TemplateRendererSkill  # add to top-of-file imports
from lottie.scaffold.schema import RenderContext, RenderInput  # add to top-of-file imports
```

Then append the plan tables and `generate` function to the bottom of the file:

```python
# (output relpath template, template name); "" template -> empty file.
_AGENT_PLAN: list[tuple[str, str]] = [
    ("AGENT.md", "agent/AGENT.md.j2"),
    ("agent.py", "agent/agent.py.j2"),
    ("schema.py", "agent/schema.py.j2"),
    ("config.yaml", "agent/config.yaml.j2"),
    ("prompts.py", "agent/prompts.py.j2"),
    ("tests/__init__.py", ""),
    ("tests/test_{name}.py", "agent/test.py.j2"),
]
_SKILL_PLAN: list[tuple[str, str]] = [
    ("SKILL.md", "skill/SKILL.md.j2"),
    ("skill.py", "skill/skill.py.j2"),
    ("schema.py", "skill/schema.py.j2"),
    ("tests/__init__.py", ""),
    ("tests/test_{name}.py", "skill/test.py.j2"),
]
_PLANS: dict[Kind, tuple[str, list[tuple[str, str]]]] = {
    "agent": ("agents", _AGENT_PLAN),
    "skill": ("skills", _SKILL_PLAN),
}


def generate(kind: Kind, name: str) -> Path:
    """Scaffold a complete `kind` module named `name`; return its directory."""
    _validate_name(name)
    class_name = _class_name(name, kind)
    root = _project_root()
    parent_dir, plan = _PLANS[kind]
    target = root / parent_dir / name
    _guard(target)

    context = RenderContext(name=name, class_name=class_name)
    files = _render_plan(plan, context, name)
    _write(target, files)
    return target


def _project_root() -> Path:
    cwd = Path.cwd()
    if not (cwd / "lottie.yaml").exists():
        raise typer.BadParameter("not a Lottie project — run `lottie init` first.")
    return cwd


def _guard(target: Path) -> None:
    if target.exists() and not target.is_dir():
        raise typer.BadParameter(f"{target} already exists as a file — choose another name.")
    if target.is_dir() and any(target.iterdir()):
        raise typer.BadParameter(f"{target} already exists and is not empty — choose another name.")


def _render_plan(
    plan: list[tuple[str, str]], context: RenderContext, name: str
) -> dict[str, str]:
    """Render every template up front so a failure writes nothing."""
    renderer = TemplateRendererSkill()
    files: dict[str, str] = {}
    for relpath, template in plan:
        out = relpath.format(name=name)
        if not template:
            files[out] = ""
            continue
        try:
            files[out] = renderer.run(RenderInput(template=template, context=context)).content
        except TemplateError as exc:
            raise typer.BadParameter(f"failed to render {template}: {exc}") from exc
    return files


def _write(target: Path, files: dict[str, str]) -> None:
    for relpath, content in files.items():
        path = target / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
```

- [ ] **Step 4: Create the CLI group**

Create `src/lottie/cli/create.py`:

```python
"""`lottie create agent|skill <name>` — scaffold a complete unit from templates."""

from __future__ import annotations

import typer

from lottie.scaffold.generator import generate

create_app = typer.Typer(help="Scaffold a new agent or skill.", no_args_is_help=True)


@create_app.command("agent")
def create_agent(name: str) -> None:
    """Scaffold a complete agent module."""
    target = generate("agent", name)
    typer.echo(f"Created agent at {target}")
    typer.echo(f"Next: implement {target / 'agent.py'} then run `pytest {target}`")


@create_app.command("skill")
def create_skill(name: str) -> None:
    """Scaffold a complete skill module."""
    target = generate("skill", name)
    typer.echo(f"Created skill at {target}")
    typer.echo(f"Next: implement {target / 'skill.py'} then run `pytest {target}`")
```

- [ ] **Step 5: Register the group**

Modify `src/lottie/cli/app.py`. Add the import beside the existing `from lottie.cli.init import init`:

```python
from lottie.cli.create import create_app
```

And add this line after the existing `app.command("init")(init)` line:

```python
app.add_typer(create_app, name="create")
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest src/lottie/cli/tests/test_create.py -v`
Expected: PASS (all create tests green).

- [ ] **Step 7: Commit**

```bash
git add src/lottie/scaffold/generator.py src/lottie/cli/create.py src/lottie/cli/app.py src/lottie/cli/tests/test_create.py
git commit -m "feat(cli): add lottie create agent/skill generator"
```

---

## Task 7: Register the unit in LOTTIE.md

**Files:**
- Modify: `src/lottie/scaffold/generator.py`
- Test: `src/lottie/cli/tests/test_create.py`

- [ ] **Step 1: Write the failing test**

Append to `src/lottie/cli/tests/test_create.py`:

```python
def test_create_updates_lottie_md(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    demo = _init_demo(tmp_path)
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "researcher"]).exit_code == 0
    assert runner.invoke(app, ["create", "skill", "cleaner"]).exit_code == 0

    md = (demo / "LOTTIE.md").read_text()
    agents_section, skills_section = md.split("## Skills")
    # Agent registered under ## Agents, placeholder replaced.
    assert "- **ResearcherAgent** — `agents/researcher/`" in agents_section
    assert "_None yet" not in agents_section
    # Skill registered under ## Skills, placeholder replaced.
    assert "- **CleanerSkill** — `skills/cleaner/`" in skills_section
    assert "_None yet" not in skills_section
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/cli/tests/test_create.py::test_create_updates_lottie_md -v`
Expected: FAIL — `LOTTIE.md` still holds the `_None yet` placeholders.

- [ ] **Step 3: Wire LOTTIE.md update into `generate`**

In `src/lottie/scaffold/generator.py`, add the call inside `generate` immediately after `_write`:

```python
    _write(target, files)
    _update_lottie_md(root, kind, class_name, f"{parent_dir}/{name}/")
    return target
```

Add this helper at the bottom of the file:

```python
def _update_lottie_md(root: Path, kind: Kind, class_name: str, location: str) -> None:
    """Register the new unit under its LOTTIE.md section. No-op if the file is absent."""
    md = root / "LOTTIE.md"
    if not md.exists():
        return
    heading = "## Agents" if kind == "agent" else "## Skills"
    entry = f"- **{class_name}** — `{location}`"
    lines = md.read_text(encoding="utf-8").splitlines()
    try:
        h = lines.index(heading)
    except ValueError:
        return
    k = h + 1
    while k < len(lines) and lines[k].strip() == "":  # skip blank lines after heading
        k += 1
    if k < len(lines) and lines[k].lstrip().startswith("_None yet"):
        lines[k] = entry  # replace the placeholder
    else:
        lines.insert(k, entry)  # prepend above the existing list
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/cli/tests/test_create.py -v`
Expected: PASS (all create tests green).

- [ ] **Step 5: Commit**

```bash
git add src/lottie/scaffold/generator.py src/lottie/cli/tests/test_create.py
git commit -m "feat(scaffold): register generated units in LOTTIE.md"
```

---

## Task 8: Generated code compiles

**Files:**
- Test: `src/lottie/cli/tests/test_create.py`

- [ ] **Step 1: Write the failing test**

Append to `src/lottie/cli/tests/test_create.py` (add `import py_compile` to the top of the file):

```python
def test_generated_python_compiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    demo = _init_demo(tmp_path)
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "researcher"]).exit_code == 0
    assert runner.invoke(app, ["create", "skill", "cleaner"]).exit_code == 0

    py_files = list((demo / "agents").rglob("*.py")) + list((demo / "skills").rglob("*.py"))
    assert py_files, "no generated .py files found"
    for f in py_files:
        py_compile.compile(str(f), doraise=True)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest src/lottie/cli/tests/test_create.py::test_generated_python_compiles -v`
Expected: PASS — every generated `.py` file is syntactically valid. (If a template has a syntax
bug, `py_compile.PyCompileError` fails the test; fix the offending template, then re-run.)

- [ ] **Step 3: Commit**

```bash
git add src/lottie/cli/tests/test_create.py
git commit -m "test(scaffold): assert generated python compiles"
```

---

## Task 9: Verification gates + manual smoke

**Files:** none (verification only)

- [ ] **Step 1: Full test suite**

Run: `uv run pytest -q`
Expected: PASS — all prior tests plus the new scaffold/cli tests green.

- [ ] **Step 2: Type check**

Run: `uv run mypy --strict src/lottie/scaffold src/lottie/cli`
Expected: `Success: no issues found`.

- [ ] **Step 3: Lint**

Run: `uv run ruff check src/lottie/scaffold src/lottie/cli`
Expected: `All checks passed!`

- [ ] **Step 4: Manual smoke — generated project is real and green**

Run:
```bash
cd "$(mktemp -d)" && \
  uv run --project /Users/cdiaz19/Documents/trae_projects/lottie-orchestrator lottie init demo && \
  cd demo && \
  uv run --project /Users/cdiaz19/Documents/trae_projects/lottie-orchestrator lottie create agent researcher && \
  uv run --project /Users/cdiaz19/Documents/trae_projects/lottie-orchestrator lottie create skill web_search && \
  uv run --project /Users/cdiaz19/Documents/trae_projects/lottie-orchestrator pytest -q
```
Expected: both `create` commands print "Created …"; `pytest` inside `demo/` runs the 3 agent +
3 skill generated tests and they all pass. Confirm `demo/LOTTIE.md` lists `ResearcherAgent` and
`WebSearchSkill`.

- [ ] **Step 5: No commit**

Verification only — nothing to commit unless a gate forced a fix (then commit that fix with the
matching `fix(...)` / `test(...)` message).

---

## Amendments (applied during execution)

Changes made after the original plan, driven by code review and the Task 9 smoke:

1. **`_validate_name` hardened** (`fix(scaffold): reject keyword/underscore names …`) — reject Python
   keywords (`class` → broken `class ClassAgent`) and leading-underscore names (`_private` →
   silently corrupted `PrivateSkill`). Added `import keyword`; conditions `name.startswith("_")`
   and `keyword.iskeyword(name)`.
2. **Renderer built once, benchmarks off** — `TemplateRendererSkill(enable_benchmarks=False)` is
   constructed once in `generate()` and passed into `_render_plan`. Stops the internal scaffolding
   skill from shelling out to `git` and writing benchmark JSONL into the user's project.
3. **Relative-path output hints** — `create.py` echoes `target.relative_to(Path.cwd())`, matching
   `init.py` style; tests assert the success/`Next:` lines.
4. **Module-level `__init__.py`** (`fix(scaffold): emit module __init__.py …`) — the original file
   plans omitted `agents/<name>/__init__.py` and `skills/<name>/__init__.py`. Without them the
   generated module isn't a package: relative imports break and the two `tests/` dirs collide
   (`ModuleNotFoundError: No module named 'tests.test_web_search'`). Added `("__init__.py", "")` as
   the first entry of both plans.
5. **Generated-project regression test** — `test_generated_project_tests_pass` scaffolds an agent +
   skill into a temp project and runs `python -m pytest` against it, asserting the 6 generated tests
   pass. `py_compile` (Task 8) only checks syntax; this catches import/packaging defects. (Defect #4
   was invisible to every in-repo test until the manual smoke — this closes that gap in CI.)
```
