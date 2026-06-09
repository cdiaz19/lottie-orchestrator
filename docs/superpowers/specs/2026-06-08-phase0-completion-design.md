# Phase 0 Completion — Design Spec

> Date: 2026-06-08
> Status: approved (brainstorming) → ready for implementation plan
> Goal: close the 5 remaining Phase 0 Deliverables Checklist items in `LOTTIE_PHASE0_SPEC.md` §8.

---

## 1. Scope

Complete all five open Phase 0 deliverables:

1. `lottie init` ships a **complete, runnable hello-world agent** (currently only scaffolds a project skeleton).
2. **CI/CD** — GitHub Actions running ruff, mypy, pytest on every push/PR.
3. **`ScaffolderAgent`** — the AI-powered `--from-desc` generator agent.
4. **Built-in skills** — `SchemaValidatorSkill` (new) joins the already-built `TemplateRendererSkill`; plus the security skills required by the code-write gate.
5. **`lottie status` auto-updates `LOTTIE.md`**.

### Decisions locked during brainstorming

| # | Decision |
|---|---|
| Scope | All 5 items (option A). |
| Pre-write gate | **Full rule-13 pipeline** — build `SecretDetectionSkill` + `CodeSecurityScanSkill` now; do not defer to Phase 1. |
| Generation model | **Hybrid** — LLM emits a structured plan that renders scaffolding, and writes the `run()` body only. |
| CI | Push + PR to `main` only (no `dev` branch). Coverage **report-only**, no hard gate. |
| Hello-world | **Complete runnable agent** bundled as a dedicated template. |
| Gate failure | **1-retry loop** feeding diagnostics back to the LLM, then hard-fail. |
| `status` → `LOTTIE.md` | `status` regenerates the registry sections AND prints the table (no separate `--write` flag). |

### Pre-existing state (verified)

- `TemplateRendererSkill` already exists as a `BaseSkill` at `src/lottie/scaffold/renderer.py` — reused as-is.
- Deps already present: `jinja2`, `litellm`, `pydantic`, `pytest-cov`.
- `src/lottie/security/` is empty (only `__init__.py`).
- `lottie create` already updates `LOTTIE.md` via `_update_lottie_md` in `scaffold/generator.py`.
- `lottie init` writes a project skeleton but **no** hello-world agent (points the user at `lottie create agent`).

---

## 2. New components & layout

### Built-in skills

Each extends `BaseSkill[Input, Output]`, is deterministic (no LLM), and has a `SKILL.md` written **before** the code (CLAUDE.md rule 3).

| Skill | Module | Behavior |
|---|---|---|
| `SchemaValidatorSkill` | `src/lottie/scaffold/validator.py` | Run `mypy --strict` + `ruff check` + a Pydantic-import sanity check over a file set. Returns pass/fail + structured diagnostics. |
| `SecretDetectionSkill` | `src/lottie/security/secret_detector.py` | `detect-secrets` library + custom regex patterns over content. Returns findings list. |
| `CodeSecurityScanSkill` | `src/lottie/security/code_scanner.py` | `bandit` over Python source. Returns findings list. (semgrep explicitly deferred — note in SKILL.md.) |
| `TemplateRendererSkill` | `src/lottie/scaffold/renderer.py` | **Already exists.** Reused unchanged. |

### Code-write gate

`src/lottie/security/write_gate.py` — single entry point `guard_and_write(files) -> GateResult`.

- Composes the rule-13 order exactly: `SecretDetectionSkill → CodeSecurityScanSkill → mypy --strict → ruff`.
- **Render-to-temp, scan, then commit.** Files are rendered to a temp location, scanned, and only moved into place on a clean pass. Any failure → abort with **zero partial writes**, diagnostics returned in `GateResult`.

### ScaffolderAgent

`src/lottie/scaffold/scaffolder_agent.py` — extends `BaseAgent`. Drives the `--from-desc` path only. Plain `lottie create agent/skill <name>` (no `--from-desc`) stays on the existing deterministic template generator, untouched.

### Dependencies to add

Runtime: `detect-secrets`, `bandit`.

---

## 3. ScaffolderAgent flow

CLI: `lottie create agent <name> --from-desc "..."` (and the `skill` variant) → `ScaffolderAgent.run(ScaffoldRequest)`.

### Schemas (`src/lottie/scaffold/schema.py`, extend existing)

- `ScaffoldRequest`: `kind` (`agent` | `skill`), `name`, `description`.
- `ScaffoldPlan` (LLM structured output): `class_name`, `input_fields[]`, `output_fields[]`, `tools[]`, `system_prompt`, `run_body` (LLM-written `run()` source).
- `GateResult`: `passed: bool`, `findings[]`, `files_written[]`.
- `ScaffoldResult`: `files_written[]`, `gate: GateResult`.

### Flow

1. **Plan** — `ScaffolderAgent` calls `LLMProvider` → `ScaffoldPlan` as structured/JSON output, validated by Pydantic (retry on parse failure). Provider resolved from `lottie.yaml`; tests inject `MockLLMProvider`.
2. **Render** — `TemplateRendererSkill` renders all files from the plan's structured fields. `run_body` is injected into the `agent.py` / `skill.py` template slot.
3. **Gate** — `guard_and_write` runs the rule-13 pipeline on the rendered files. Fail → abort, nothing written, diagnostics surfaced.
4. **Retry** — on `mypy`/`ruff` failure, feed diagnostics back to the LLM **once** to repair `run_body`, re-render, re-gate. Still failing → hard-fail with the full report.
5. **Write + register** — on pass, commit files and update `LOTTIE.md` (reuse the shared registry writer from §4, item #5).

### Tests (MockLLM)

- Canned `ScaffoldPlan` → assert gate runs in rule-13 order.
- Clean plan → files written, `LOTTIE.md` updated.
- Plan with a planted secret → aborts, **no write**.
- Plan with a bandit-flagged construct → aborts, no write.
- Plan whose `run_body` fails mypy once then passes → retry path writes; fails twice → hard-fail, no write.

---

## 4. Smaller items

### #1 — Hello-world agent

Bundled complete template at `src/lottie/scaffold/templates/hello/`: `AGENT.md`, `agent.py`, `schema.py`, `config.yaml`, `prompts.py`, `tests/test_hello.py`.

- `agent.py` has a real `run()` that greets the input via `LLMProvider`.
- Tests use `MockLLMProvider`.
- `lottie init` writes it into `agents/hello/`.
- Acceptance: `lottie run hello --input '{"name":"x"}'` works end-to-end.

### #5 — `lottie status` → `LOTTIE.md`

- Refactor `_update_lottie_md` (currently in `scaffold/generator.py`) into a reusable full-registry writer at `src/lottie/scaffold/lottie_md.py`.
- `lottie status` rebuilds the Agents/Skills sections from discovery, then prints the status table.
- Writer is idempotent — repeated runs produce identical output.

### #2 — CI

`.github/workflows/ci.yml`:

- Triggers: `push` and `pull_request` targeting `main`.
- Runner: single job, Python 3.12, `uv`-based.
- Steps: `ruff check` → `mypy --strict` → `pytest` → coverage (report only, **no hard gate**).

---

## 5. Build order

Dependency-first. Items 4/5/6 are independent of 1–3 and can land in any order.

1. `SchemaValidatorSkill` + `SecretDetectionSkill` + `CodeSecurityScanSkill` (add deps `detect-secrets`, `bandit`). Each TDD with unit tests, no LLM.
2. `write_gate.guard_and_write` — composes step 1 in rule-13 order; render-to-temp-then-commit.
3. `ScaffolderAgent` + schemas + `--from-desc` CLI wiring — uses steps 1+2; MockLLM tests; 1-retry loop.
4. Hello-world template + `init` wiring.
5. `status` → `LOTTIE.md` writer refactor.
6. CI workflow.

Per-step rules: write `SKILL.md` / `AGENT.md` before code (rule 3); `MockLLMProvider` only in unit/integration tests, never real LLM (rule 5); every file passes `mypy --strict` (rule 6).

---

## 6. Acceptance — Phase 0 checklist closed

- [x→] `lottie init` creates a valid project **with a working hello-world agent** that `lottie run` executes end-to-end.
- [x→] CI/CD: ruff, mypy, pytest on every push/PR to `main`.
- [x→] `ScaffolderAgent` generates agents/skills from `--from-desc`, hybrid model, behind the rule-13 gate.
- [x→] `SchemaValidatorSkill` built (`TemplateRendererSkill` already present); security skills `SecretDetectionSkill` + `CodeSecurityScanSkill` built.
- [x→] `LOTTIE.md` auto-updated by `lottie status`.

All five Phase 0 Deliverables Checklist items move from open to done.
