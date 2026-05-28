# Design — `lottie init` CLI

> Phase 0 · scaffold a new Lottie project skeleton.
> Spec ref: `LOTTIE_PHASE0_SPEC.md` §3 (Project commands), §4 (file structure).

## Goal

Provide `lottie init <name>` — the first CLI command — which scaffolds a new
Lottie project skeleton (directory tree, `lottie.yaml`, `LOTTIE.md`, policies,
`.gitignore`). No agent is generated yet; the hello-world agent is deferred
until the `lottie create agent` generator exists.

## Scope

**In scope**
- Wire the `lottie` console entry point (Typer app) — prerequisite for any CLI command.
- `lottie init <name> [--here]` command + its scaffold templates.
- Unit tests (no LLM) via Typer `CliRunner`.

**Out of scope (deferred)**
- hello-world agent generation — lands with `lottie create agent`.
- Generating a `pyproject.toml` in the scaffolded project — the framework
  (`lottie-orchestrator`) is the dependency; user projects are config + agents/skills.
- `.lottie/` runtime dir — created on first run, not at init.
- Other commands (`status`, `doctor`, `create`, `run`) — separate work items.

## Entry point

Add to project `pyproject.toml`:

```toml
[project.scripts]
lottie = "lottie.cli:app"
```

- `src/lottie/cli/app.py` — module-level `app = typer.Typer()`. Subcommands register here.
- `src/lottie/cli/__init__.py` — re-exports `app` so `lottie.cli:app` resolves.
- `init` is the first registered subcommand. `status` / `doctor` / `create` / `run`
  attach to the same `app` in later work items.

## Command behavior

`lottie init <name> [--here]`

| Mode | Target | Guard (refuse, exit ≠ 0, no writes) |
|---|---|---|
| default | `./<name>/` | dir exists and is non-empty |
| `--here` | current working dir | `lottie.yaml` already present in cwd |

- `<name>` is always recorded as the project name inside `lottie.yaml` and `LOTTIE.md`.
- Guard runs **before any write** — no partial scaffolds on error.
- Errors raise `typer.BadParameter` (clean message + non-zero exit).
- On success: print created path + a short "next steps" hint.

## Scaffold tree

```
<target>/
  lottie.yaml
  LOTTIE.md
  .gitignore
  agents/__init__.py
  skills/__init__.py
  policies/base.yaml
  knowledge/global/.gitkeep
  knowledge/platform/.gitkeep
  knowledge/project/.gitkeep
  knowledge/memory/.gitkeep
  knowledge/draft/.gitkeep
```

### File contents

**`lottie.yaml`** — providers from personal defaults, single base policy, registry paths:
```yaml
project: <name>
providers:
  default: anthropic/claude-sonnet-4-6
  fallback: openai/gpt-4o
policies:
  - base
registry:
  agents: agents/
  skills: skills/
```

**`LOTTIE.md`** — project doc header:
```markdown
# <name>

> A Lottie project. This file is read automatically by all AI tools.

## Agents
_None yet — scaffold one with `lottie create agent <name>`._

## Skills
_None yet — scaffold one with `lottie create skill <name>`._
```

**`.gitignore`**:
```gitignore
# Lottie runtime
.lottie/

# Private AI context
.private-journey/

# Personal Claude Code settings
.claude/settings.local.json

# Python
__pycache__/
.venv/
```

**`policies/base.yaml`** — allow/deny/escalate skeleton with empty rule lists:
```yaml
# Base governance policy. Rules: allow / deny / escalate.
name: base
allow: []
deny: []
escalate: []
```

**`agents/__init__.py` / `skills/__init__.py`** — module docstring placeholder noting
these packages auto-discover registered agents/skills (discovery logic is a later work item).

**`.gitkeep`** — empty, keeps the five knowledge-layer dirs tracked in git.

## Module structure

- `cli/templates.py` — scaffold contents as module-level string constants. The two
  templates needing the project name use `str.format(name=...)`; the rest are static.
- `cli/init.py` — `init` command: resolve target dir → guard → write tree.
- `cli/app.py` — thin Typer app; imports and registers `init`.

Keeping templates and command logic separate keeps `app.py` minimal and lets the
generator work item later reuse template strings.

## Testing (TDD, no LLM)

`src/lottie/cli/tests/test_init.py` — Typer `CliRunner`, all writes in `tmp_path`:

1. `init <name>` creates `./<name>/` with the full tree and all files.
2. `--here` scaffolds into cwd (no extra subdir).
3. `lottie.yaml` records the given project name.
4. Refuses a non-empty `./<name>/` — exit ≠ 0, nothing written.
5. Refuses `--here` when cwd already has `lottie.yaml` — exit ≠ 0, nothing written.
6. `.gitignore` contains the runtime + private-context paths.
7. `policies/base.yaml` parses as YAML with `allow`/`deny`/`escalate` keys.

## Verification gates (CLAUDE.md)

- `pytest` — all green
- `mypy --strict src/lottie/cli` — clean
- `ruff check src/lottie/cli` — clean
- Manual smoke: `uv run lottie init demo` in a temp dir produces a valid tree.
