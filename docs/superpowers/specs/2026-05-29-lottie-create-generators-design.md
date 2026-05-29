# Design — `lottie create agent/skill` generators

> Phase 0 · scaffold a complete agent or skill from templates.
> Spec ref: `LOTTIE_PHASE0_SPEC.md` §5 (file standards), §6 (generator), §9 (testing).

## Goal

Provide `lottie create agent <name>` and `lottie create skill <name>` — the
self-extending core of Lottie. Each command scaffolds a complete, working
agent or skill module (all required files) from Jinja2 templates, guards
against clobbering, and registers the new unit in `LOTTIE.md`.

## Scope

**In scope**
- `create` Typer sub-group with `agent` and `skill` subcommands.
- `TemplateRendererSkill` — a real `BaseSkill` rendering Jinja2 templates with a typed contract.
- `lottie.scaffold` package: renderer, generator orchestrator, `.j2` templates.
- Name validation, clobber guard, `LOTTIE.md` registration.
- Generated stubs that compile, pass `mypy --strict`, and ship 3 green tests each.
- Unit tests (no real LLM) for the renderer, the generator, and the CLI surface.

**Out of scope (deferred)**
- `--from-desc "..."` AI generation + `ScaffolderAgent` — its own work item; `lottie.scaffold`
  is laid out so the agent slots in alongside the renderer.
- `SchemaValidatorSkill` (running `mypy`/`ruff` on generated files as a gate). This item uses a
  lightweight `py_compile` check instead; the full validator lands later.
- Reading `registry:` paths from `lottie.yaml` (no PyYAML dep yet) — target dirs are the
  hardcoded `agents/` and `skills/` defaults that `lottie init` writes.
- `evals/eval_<name>.py` — optional per spec §6; not generated.

## Command surface

`create` is a Typer sub-group registered on the root app:

| Command | Action |
|---|---|
| `lottie create agent <name>` | scaffold `agents/<name>/` (all required files) |
| `lottie create skill <name>` | scaffold `skills/<name>/` (all required files) |

- `cli/create.py` builds `create_app = typer.Typer()`, registers `agent`/`skill`, and the root
  `app.add_typer(create_app, name="create")`.
- Each subcommand is a thin wrapper delegating to `lottie.scaffold.generator`.
- `--from-desc` is **not** wired this item.

## Module layout

```
src/lottie/scaffold/
  __init__.py
  schema.py        # RenderContext, RenderInput, RenderOutput (Pydantic v2)
  renderer.py      # TemplateRendererSkill(BaseSkill[RenderInput, RenderOutput])
  generator.py     # orchestrator: validate -> resolve -> guard -> render -> write -> update LOTTIE.md
  templates/
    agent/  AGENT.md.j2  agent.py.j2  schema.py.j2  config.yaml.j2  prompts.py.j2  test.py.j2
    skill/  SKILL.md.j2  skill.py.j2  schema.py.j2  test.py.j2
  tests/
    __init__.py  test_renderer.py  test_generator.py
src/lottie/cli/create.py            # thin Typer `create` group
src/lottie/cli/tests/test_create.py # CliRunner end-to-end
```

`lottie.scaffold` is the cohesive home for all generation logic. The future `ScaffolderAgent` and
`SchemaValidatorSkill` land here and reuse `TemplateRendererSkill`.

## TemplateRendererSkill

`BaseSkill[RenderInput, RenderOutput]`. A Jinja2 `Environment` with a loader rooted at
`scaffold/templates/`, `undefined=StrictUndefined` (missing vars fail loudly), and
`keep_trailing_newline=True`.

**Template loading — must survive an installed wheel, not just editable dev.** The `.j2` files
are package *data*, not importable modules. Use `jinja2.PackageLoader("lottie.scaffold",
"templates")`, which resolves via `importlib` and works regardless of install layout — *provided
the build ships the files*. With the current `hatchling` wheel target (`packages =
["src/lottie"]`) non-`.py` files under the package tree are included by default, so `.j2` files
ship. To make this explicit and guard against regressions, add a force-include glob (see
Dependency section). Avoid `FileSystemLoader(Path(__file__).parent / ...)`: `__file__`-relative
paths break under zip-imported or relocated installs. A renderer test exercises this by rendering
a real on-disk template, so a packaging regression fails CI.

```python
class RenderContext(BaseModel):
    name: str                                   # snake_case module/dir name
    class_name: str                             # PascalCase + "Agent"/"Skill"
    provider: str = "anthropic/claude-sonnet-4-6"

class RenderInput(BaseModel):
    template: str                               # e.g. "agent/agent.py.j2"
    context: RenderContext

class RenderOutput(BaseModel):
    content: str
```

- `_execute`: load `data.template`, `render(**data.context.model_dump())`, return `RenderOutput`.
- Unknown template or undefined var raises; the generator catches and re-raises as
  `typer.BadParameter` for a clean CLI message.
- Honors CLAUDE.md rule 2: the skill boundary is fully typed; the dict passed to Jinja is internal.

## Generator flow (`generator.py`)

A single `generate(kind: Literal["agent","skill"], name: str)` entry point:

1. **Validate name** — single path segment, valid Python identifier, no path separators / dots /
   leading uppercase / empties. Mirrors `cli/init.py::_validate_name`.
2. **Derive `class_name`** — `snake_to_pascal(name) + {"agent":"Agent","skill":"Skill"}[kind]`.
   e.g. `web_search` -> `WebSearchSkill`.
3. **Resolve project root** — cwd must contain `lottie.yaml`, else
   `typer.BadParameter("not a Lottie project — run lottie init first")`.
4. **Guard** — target is `<root>/{agents|skills}/<name>/`; refuse if it exists and is non-empty.
   Build the full rendered file map in memory **before** any write — no partial scaffolds on error.
5. **Render + write** — for each `(relpath, template)` in the kind's file plan, call
   `TemplateRendererSkill.run(...)`; create dirs and write the tree (incl. `tests/__init__.py`).
6. **Update LOTTIE.md** — under `## Agents` / `## Skills`, replace the `_None yet …_` placeholder
   with `- **<ClassName>** — \`{agents|skills}/<name>/\``; if entries already exist, append a line;
   if `LOTTIE.md` is absent, skip silently (do not fail the scaffold).
7. **Echo** the created path and a next-step hint.

### File plans

**Agent** (`agents/<name>/`):
| Output | Template |
|---|---|
| `AGENT.md` | `agent/AGENT.md.j2` |
| `agent.py` | `agent/agent.py.j2` |
| `schema.py` | `agent/schema.py.j2` |
| `config.yaml` | `agent/config.yaml.j2` |
| `prompts.py` | `agent/prompts.py.j2` |
| `tests/__init__.py` | (empty) |
| `tests/test_<name>.py` | `agent/test.py.j2` |

**Skill** (`skills/<name>/`):
| Output | Template |
|---|---|
| `SKILL.md` | `skill/SKILL.md.j2` |
| `skill.py` | `skill/skill.py.j2` |
| `schema.py` | `skill/schema.py.j2` |
| `tests/__init__.py` | (empty) |
| `tests/test_<name>.py` | `skill/test.py.j2` |

## Generated stubs

Generated code is the contract: it must compile, pass `mypy --strict`, and its 3 generated tests
must pass immediately after scaffolding.

**Agent**
- `schema.py`: `Input{query: str}`, `Output{result: str}` (Pydantic v2).
- `agent.py`: `_execute` does
  `resp = self.complete([Message(role="user", content=data.query)]); return Output(result=resp.content)`.
- `prompts.py`: a `SYSTEM_PROMPT: str` constant.
- `config.yaml`: provider + `capabilities: []` + `policies: [base]` (spec §5 / security §12 shape).
- `tests/test_<name>.py` — 3 cases via `MockLLMProvider`: normal flow (result echoes the mock
  response), records exactly one LLM call, edge (empty query).

**Skill**
- `schema.py`: `Input{text: str}`, `Output{result: str}`.
- `skill.py`: deterministic `_execute` returning `Output(result=data.text)`.
- `tests/test_<name>.py` — 3 cases, no LLM: happy path, edge (empty string), error
  (`Input(text=123)` raises `ValidationError`).

Both get `AGENT.md` / `SKILL.md` filled from the spec §5 doc templates with name/class/provider
substituted.

## Testing (TDD, no real LLM)

**`scaffold/tests/test_renderer.py`**
- Renders a real template with a `RenderContext` → output contains `class_name`.
- `StrictUndefined` raises when a template references an unset var.
- Unknown template name raises.

**`scaffold/tests/test_generator.py`** and **`cli/tests/test_create.py`** (CliRunner, in `tmp_path`
with a scaffolded project):
- `create agent web_search` → full file tree exists; `agent.py` contains
  `class WebSearchAgent(BaseAgent)`.
- `create skill cleaner` → full tree; `skill.py` contains `class CleanerSkill(BaseSkill)`.
- `LOTTIE.md` gains the registration entry.
- Guard refuses an existing non-empty target — exit ≠ 0, nothing overwritten.
- Refuses when cwd has no `lottie.yaml` — exit ≠ 0, nothing written.
- Invalid names rejected (parametrized) — exit ≠ 0, nothing written.
- `py_compile` over every generated `.py` file — guarantees syntactic validity (lightweight proxy
  for "it compiles"; full `mypy`/`pytest`-on-generated deferred to `SchemaValidatorSkill`).

## Dependency

Add `jinja2` to `[project].dependencies` in `pyproject.toml`.

**Package the templates explicitly.** Add a force-include so the `.j2` files always land in the
wheel even if the default-inclusion behavior changes:

```toml
[tool.hatch.build.targets.wheel.force-include]
"src/lottie/scaffold/templates" = "lottie/scaffold/templates"
```

Verify with `uv build` + inspecting the wheel (or `python -c "from jinja2 import PackageLoader;
PackageLoader('lottie.scaffold','templates')"` against an installed build) that templates resolve.

## Verification gates (CLAUDE.md)

- `uv run pytest` — all green.
- `uv run mypy --strict src/lottie/scaffold src/lottie/cli` — clean.
- `uv run ruff check src/lottie/scaffold src/lottie/cli` — clean.
- Manual smoke: `lottie init demo && cd demo && lottie create agent researcher && lottie create
  skill web_search`, then `uv run pytest` inside `demo/` — generated tests green.
