# `lottie list` / `lottie inspect` — Design

> Date: 2026-06-01
> Phase: 0 — Foundations (Registry CLI)
> Status: approved

## Goal

Add the registry-query commands from the Phase 0 spec, completing the CLI
surface alongside `run` / `status` / `doctor`:

- `lottie list agents` — registered agents with their provider
- `lottie list skills` — registered skills with their input/output types
- `lottie inspect agent <name>` — full config, schema, system prompt
- `lottie inspect skill <name>` — SKILL.md presence, schema

Both build on the shipped `lottie.project` discovery layer. No new runtime
dependencies.

## Scope decisions

- **Omit columns with no backing data.** The Phase 0 spec lists `benchmark
  score`, `last-run`, and `version` for `list`. There is no benchmark store and
  no run-history store yet, so those columns are omitted now and added when
  `lottie benchmark` and run history ship. No `—`-filled placeholder columns.
- **`inspect` imports user code (like `run`).** `inspect` is a single-target
  deep view, so it imports `schema.py` / `prompts.py` to render resolved
  Pydantic fields and the actual system prompt. A broken unit raises
  `typer.BadParameter` cleanly.
- **`list` stays robust.** `list agents` is pure discovery (no import).
  `list skills` must import each skill's `schema.py` to name its I/O types; this
  is done per-skill in `try/except` so one broken skill degrades to `—` / `—`
  rather than crashing the whole list.
- **`inspect skill` is supported** (not only `inspect agent`) — cheap symmetry
  with `list skills`.

## Architecture

New module `src/lottie/cli/registry.py` exposing two Typer apps, registered in
`src/lottie/cli/app.py`:

- `list_app` (sub-Typer) → `agents`, `skills` commands → `app.add_typer(list_app, name="list")`
- `inspect_app` (sub-Typer) → `agent`, `skill` commands → `app.add_typer(inspect_app, name="inspect")`

Reuses: `find_project_root`, `load_lottie_config`, `load_agent_config`
(`lottie.project.config`); `discover_agents`, `discover_skills`,
`UnitInfo` (`lottie.project.discovery`). Output via `rich` `Console` / `Table` /
`Panel`, matching `status.py`.

## Discovery-layer changes (`lottie/project/discovery.py`)

The prefixed/legacy model-detection logic currently lives inside
`load_input_model` and is hardcoded to `agents.{name}.schema`. Generalize it:

- Extract a private helper that, given a dotted module path and a suffix
  (`"Input"` / `"Output"`), returns the matching `BaseModel` subclass (legacy
  bare `Input`/`Output` name, or exactly one `<Class><Suffix>` class).
- `load_schema_models(root: Path, kind: Literal["agent","skill"], name: str) -> tuple[type[BaseModel], type[BaseModel]]`
  — imports `{kind}s.{name}.schema` and returns `(Input, Output)`.
- `load_system_prompt(root: Path, name: str) -> str | None` — imports
  `agents.{name}.prompts` and returns `SYSTEM_PROMPT`, or `None` if the module /
  attribute is absent.
- `load_input_model` is kept as a thin wrapper over `load_schema_models(root,
  "agent", name)[0]` so `run` is unaffected.

Errors: missing module → `typer.BadParameter("<kind> '<name>' not found")`;
import failure → `typer.BadParameter("<kind> '<name>' failed to import: …")`;
schema present but no Input/Output class → existing descriptive
`typer.BadParameter`. Mirrors current `_import_unit_module` behavior.

## Command behavior

### `lottie list agents`
1. `root = find_project_root()`
2. `units = discover_agents(root)`
3. Empty → `console.print("_No agents yet._")`.
4. Else Rich table, columns `name`, `provider` (`unit.provider or "—"`).

### `lottie list skills`
1. `root = find_project_root()`; `units = discover_skills(root)`.
2. Empty → `_No skills yet._`.
3. Else table columns `name`, `input`, `output`. For each skill, attempt
   `load_schema_models(root, "skill", name)`; on success show the two class
   `__name__`s, on any exception show `—` / `—`.

### `lottie inspect agent <name>`
1. `root = find_project_root()`.
2. Validate the agent dir exists in `discover_agents(root)`; unknown →
   `typer.BadParameter`.
3. Load `config.yaml` via `load_agent_config`. Render: provider,
   `model_params`, `capabilities` (or `—`), `policies`.
4. `load_schema_models(root, "agent", name)` → render Input + Output field
   `name: type` lines.
5. `load_system_prompt(root, name)` → render verbatim, or `—`.
6. Print as Rich `Panel`(s) titled by section.

### `lottie inspect skill <name>`
1. Validate dir exists in `discover_skills(root)`; unknown → `typer.BadParameter`.
2. `load_schema_models(root, "skill", name)` → Input + Output fields.
3. Note `SKILL.md` presence. (Skills have no provider / system prompt.)

## Testing (tests-first)

New `src/lottie/cli/tests/test_registry.py`, plus discovery-loader tests in
`src/lottie/project/tests/test_discovery.py`. Use Typer `CliRunner` against the
real `app`, in a tmp project scaffolded via the existing
`generate("agent"/"skill", name)` path (or the established fixture used by
`test_status`/`test_run`). No real LLM — schemas are pure Pydantic; nothing here
calls a provider.

Cases:
- `list agents` / `list skills` empty → "No … yet" message, exit 0.
- `list agents` populated → row per agent with provider.
- `list skills` populated → row with Input/Output class names.
- `list skills` with a skill whose `schema.py` raises on import → that row shows
  `—` / `—`, other rows intact, exit 0.
- `inspect agent <name>` → output contains provider, capabilities, Input/Output
  field, system-prompt text.
- `inspect skill <name>` → output contains Input/Output fields.
- `inspect agent <unknown>` / `inspect skill <unknown>` → non-zero exit,
  BadParameter message.
- Any command outside a Lottie project (no `lottie.yaml`) → BadParameter.
- Discovery unit: `load_schema_models` for agent + skill, prefixed and legacy
  class names; `load_system_prompt` present / absent.

## Out of scope

- Benchmark score, last-run, version columns (no store yet).
- `lottie inspect` performance history.
- JSON / machine-readable output (`--json`) — add later if an integration needs it.
- `mypy --strict` and `ruff` must stay clean; no `Any` without justification.
