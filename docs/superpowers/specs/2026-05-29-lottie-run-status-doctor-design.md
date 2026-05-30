# Design — `lottie run` / `status` / `doctor`

> Phase 0 · run an agent end-to-end; inspect and health-check a project.
> Spec ref: `LOTTIE_PHASE0_SPEC.md` §3 (Project + Running commands).

## Goal

Add the three remaining Phase-0 project commands:
- `lottie run <name> [--input JSON] [--provider P]` — load and execute an agent end-to-end.
- `lottie status` — show project config, registered agents/skills, knowledge size.
- `lottie doctor` — check environment health (Python, deps, API keys, project validity).

All three build on a new shared `src/lottie/project/` layer (project resolution, typed config,
unit discovery/loading) that `serve`/`benchmark`/MCP can reuse later.

## Scope

**In scope**
- `src/lottie/project/` package: `config.py` (root resolution + typed configs + YAML) and
  `discovery.py` (metadata discovery without import; class loading with import for `run`).
- `lottie.llm.build_provider(model)` — single provider construction point.
- `cli/run.py`, `cli/status.py`, `cli/doctor.py` (thin) + registration in `cli/app.py`.
- `pyyaml` as a direct dependency.
- Unit tests (no real LLM) for the project layer and all three commands.

**Out of scope (deferred)**
- Streaming output — `BaseAgent` is sync-first; `run` prints the final result. Streaming lands with
  the async wrapper.
- Live provider reachability ping in `doctor` — needs network, untestable offline. `doctor` checks
  API-key *presence* only.
- Interactive field-by-field input prompting — `run` takes `--input` JSON only.
- Running skills directly (`lottie run` is agent-only per spec §3).
- `lottie list` / `inspect` (spec §3 Registry) — separate work item.

## The shared layer — `src/lottie/project/`

### `project/config.py`
- `find_project_root() -> Path` — walk up from `Path.cwd()` looking for `lottie.yaml`; raise
  `typer.BadParameter("not a Lottie project — run \`lottie init\` first.")` if none found.
- Typed models (Pydantic v2; CLAUDE.md rule 2):
  ```python
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
  ```
- `load_lottie_config(root: Path) -> LottieConfig` — `yaml.safe_load(lottie.yaml)` → `model_validate`.
- `load_agent_config(unit_dir: Path) -> AgentConfig` — same for `<unit_dir>/config.yaml`.
- Malformed YAML or schema-invalid config → `typer.BadParameter` with the file path and reason.

### `project/discovery.py`
Discovery (metadata, **no import**) is separate from loading (**imports user code**) so a broken
agent can't crash `status`/`doctor`.

Discovery uses the conventional `agents/` and `skills/` directories (matching `lottie create`).
The `registry` paths in `lottie.yaml` are parsed into `LottieConfig` but not yet used for path
resolution — wiring them in is deferred until a project needs non-default locations.
- `UnitInfo(BaseModel)` — `name: str`, `kind: Literal["agent", "skill"]`, `provider: str | None`,
  `path: Path`.
- `discover_agents(root: Path) -> list[UnitInfo]` — list `<root>/agents/*/` dirs containing
  `agent.py`; read each `config.yaml` for `provider` (None if missing/unreadable). Sorted by name.
- `discover_skills(root: Path) -> list[UnitInfo]` — same for `<root>/skills/*/` with `skill.py`;
  skills have no provider (`None`).
- `load_agent_class(root: Path, name: str) -> type[BaseAgent]` — ensure `str(root)` is on
  `sys.path`, `importlib.import_module(f"agents.{name}.agent")`, return the single class that is a
  `BaseAgent` subclass **and defined in that module** (`obj.__module__ == module.__name__`). Raise
  `typer.BadParameter` on zero or multiple matches, or `ModuleNotFoundError` → friendly
  "agent '<name>' not found".
- `load_input_model(root: Path, name: str) -> type[BaseModel]` — import `agents.<name>.schema`,
  return its `Input` attribute.
- `required_fields(model: type[BaseModel]) -> list[str]` — field names with no default (used by
  `run` to message when `--input` is missing).

## `lottie run` (`cli/run.py`)

`lottie run <name> [--input JSON] [--provider P]`

1. `root = find_project_root()`; `cfg = load_agent_config(root / "agents" / name)`
   (if the dir is missing → `BadParameter` "agent '<name>' not found").
2. `model = provider_override or cfg.provider`.
3. `provider = build_provider(model)`.
4. `Input = load_input_model(root, name)`:
   - `--input` given → `Input.model_validate_json(raw)`; `ValidationError`/JSON error → exit 1 with
     the validation message.
   - absent + `required_fields(Input)` non-empty → `BadParameter` listing the required fields.
   - absent + all-optional → `Input()`.
5. `Cls = load_agent_class(root, name)`; `agent = Cls(llm=provider)`; `result = agent.run(input_obj)`.
6. Print `result.model_dump_json(indent=2)`.

Errors (agent not found, bad JSON, missing required input, provider/LLM failure) exit non-zero with
a clear message — never a raw traceback. `run` makes a real LLM call in normal use; tests
monkeypatch `litellm.completion`.

## `lottie status` (`cli/status.py`)

No LLM, no user-code import. `rich` output:
- Header: project name; providers (default + fallback); policies.
- Agents table: `name` · `provider`. Skills table: `name` · `provider` (provider blank for skills).
- Knowledge: count of files per `knowledge/<layer>/` dir (excluding `.gitkeep`), if `knowledge/`
  exists.
- Empty registry → "_No agents yet._" / "_No skills yet._" lines.

## `lottie doctor` (`cli/doctor.py`)

Environment health; `rich` ✓/✗ list; exit 1 if any check fails:
- Python ≥ 3.12.
- Core deps importable: `litellm`, `jinja2`, `pydantic`, `yaml`.
- In a Lottie project: `lottie.yaml` present and parses (✗ but not fatal if run outside a project —
  reported as "not in a Lottie project").
- API keys — for each provider referenced in `lottie.yaml` (default + fallback), map prefix → env
  var and check presence:
  - `anthropic` → `ANTHROPIC_API_KEY`
  - `openai` → `OPENAI_API_KEY`
  - `ollama` → no key needed (✓)
  - unknown prefix → warn "set its API key manually" (does not fail the run).
- **No live network ping** — presence only. Documented limitation.

## Provider factory + dependency

- `lottie.llm.build_provider(model: str) -> LLMProvider` — returns `LiteLLMProvider(model)`. Single
  construction point; the seam tests monkeypatch `litellm.completion` beneath. Exported from
  `lottie.llm`.
- Add `pyyaml` to `[project].dependencies` (present transitively today; make it explicit).

## Module structure

| File | Responsibility |
|---|---|
| `src/lottie/project/__init__.py` | re-export the public surface |
| `src/lottie/project/config.py` | root resolution + typed configs + YAML load |
| `src/lottie/project/discovery.py` | discover (no import) + load (import) units |
| `src/lottie/llm/__init__.py` (modify) | add/export `build_provider` |
| `src/lottie/cli/run.py` | `run` command |
| `src/lottie/cli/status.py` | `status` command |
| `src/lottie/cli/doctor.py` | `doctor` command |
| `src/lottie/cli/app.py` (modify) | register the three commands |
| `pyproject.toml` (modify) | add `pyyaml` |

## Testing (TDD, no real LLM)

- `project/tests/test_config.py` — `find_project_root` (found/not-found), parse valid/missing/
  malformed `lottie.yaml` + `config.yaml`.
- `project/tests/test_discovery.py` — discover agents/skills from a scaffolded tmp project;
  `load_agent_class` success / zero-subclass / multiple-subclass; `required_fields`.
- `cli/tests/test_run.py` — scaffold agent, **monkeypatch `litellm.completion`** → canned content:
  `run agent --input '{"query":"hi"}'` → output JSON contains the content; `--provider` override
  reaches litellm (assert the model passed); errors: unknown agent, malformed JSON, missing
  required input.
- `cli/tests/test_status.py` — scaffold init + `create agent`/`create skill` → output lists names,
  providers, knowledge counts; empty-project case.
- `cli/tests/test_doctor.py` — monkeypatch env (key present/absent) → matching ✓/✗ + exit code;
  outside-project case.

All CLI tests use `typer.testing.CliRunner` in `tmp_path`, scaffolding real projects via the
existing `init`/`create` commands.

## Verification gates (CLAUDE.md)

- `uv run pytest` — green.
- `uv run mypy --strict src/lottie` — clean.
- `uv run ruff check src/lottie` — clean.
- Manual smoke: in a scaffolded project with an agent, `ANTHROPIC_API_KEY` set →
  `lottie status`, `lottie doctor`, and `lottie run <agent> --input '{"query":"..."}'` behave.
