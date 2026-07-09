# V2 S3a — Trace→Skill Distillation (generate + run) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn an agent's accumulated learned lessons into a reusable, parameterized **prompt-template skill** — no generated code — written as a **draft** (not yet callable), with provenance; plus a generic runner that executes any such template. Human promotion to callable is S3b.

**Architecture:** A new `lottie.distill` package: typed `DistilledSkillSpec`/`DistillProvenance`/`TemplateInput`/`TemplateOutput`; pure `generate` helpers (build the distill prompt, extract `{slot}` names deterministically by regex, fill a template); a generic `TemplateRunnerSkill(BaseSkill)` that fills a template from `{slots}` and calls the LLM; on-disk `save_draft`/`load_spec` under `skills/draft/<name>/`. A `lottie distill <agent>` CLI recalls the agent's SEMANTIC memory notes, asks the LLM to synthesize one template, secret-scans it (templates are legitimately instructional, so injection-scanning is deferred to the human promotion gate in S3b), and writes the draft. No promotion, no versioning-bump, no capability wiring — those are S3b.

**Tech Stack:** Python 3.12+, Pydantic v2, PyYAML, `uv`, `pytest`, `mypy --strict`, `ruff`.

## Global Constraints

- **Rule 1:** no LLM SDK; LLM calls go through `LLMProvider`. **Rule 2:** typed models cross boundaries. **Rule 3:** every skill needs `SKILL.md` — the draft writes one. **Rule 5:** unit tests use `MockLLMProvider`/`MockMemoryClient`. **Rule 6 / 7b:** `mypy --strict` (no `Any`, no `# type: ignore`) + `ruff` + `pytest --all-extras` green before push. **Rule 7:** conventional commits.
- **D2 — no codegen:** a distilled skill is a prompt template + declared slots, executed by the one generic `TemplateRunnerSkill`. NO LLM-authored Python is ever written or executed. Slots are extracted deterministically from the template text (regex), not authored.
- **Draft-only / not callable:** `lottie distill` writes `skills/draft/<name>/` (`template.yaml` + `provenance.yaml` + `SKILL.md`) — no `skill.py`, so `discover_skills` never picks it up. It cannot run until a human promotes it (S3b).
- **Content gate:** the generated template is scanned for **secrets** (`SecretDetectionSkill`, fail-closed → no draft written). It is deliberately NOT injection-scanned: a prompt template is by nature instructional, so the injection scanner would false-reject legitimate templates. Injection/malice review is the human promotion gate (S3b) — the trust boundary for making a template callable.
- **Provenance:** the draft records `source_agent`, `namespace`, and the `source_run_ids` of the notes it was distilled from, plus a `version`.
- **Acyclic imports:** `distill.schema`/`distill.generate` are pure (pydantic + `lottie.llm` Message + stdlib). `distill.runner` imports `lottie.core` (BaseSkill) + `distill.schema`/`generate`. `distill.io` imports pydantic + yaml + `distill.schema`. No `distill → project/cli` edges (the CLI imports distill, not vice-versa).
- **Scope:** S3a = format + runner + generate + `lottie distill` (→ draft) ONLY. NO promotion (`distill review`/promote), NO callable registration, NO capability declaration, NO re-distill version bump (all S3b). NO benchmark/harness.

---

## File Structure

- `src/lottie/distill/__init__.py` — **create**: package exports.
- `src/lottie/distill/schema.py` — **create**: `DistilledSkillSpec`, `DistillProvenance`, `TemplateInput`, `TemplateOutput`.
- `src/lottie/distill/generate.py` — **create**: `DISTILL_SYSTEM_PROMPT`, `build_distill_prompt`, `extract_slots`, `fill_template`.
- `src/lottie/distill/runner.py` — **create**: `TemplateRunnerSkill`.
- `src/lottie/distill/io.py` — **create**: `save_draft`, `load_spec`.
- `src/lottie/cli/distill.py` — **create**: `lottie distill <agent>`.
- `src/lottie/cli/app.py` — **modify**: register the command.
- `CLAUDE.md` — **modify**: one-line note.
- Tests: `src/lottie/distill/tests/test_schema_io.py`, `test_generate.py`, `test_runner.py`; `src/lottie/cli/tests/test_distill_cli.py`.

---

## Task 1: distill schema + on-disk draft I/O

**Files:**
- Create: `src/lottie/distill/__init__.py`, `src/lottie/distill/schema.py`, `src/lottie/distill/io.py`
- Test: `src/lottie/distill/tests/test_schema_io.py`

**Interfaces:**
- Produces: `DistilledSkillSpec(name: str, template: str, slots: list[str] = [], version: str = "0.1.0")`; `DistillProvenance(source_agent: str, namespace: str, source_run_ids: list[str] = [], version: str = "0.1.0")`; `TemplateInput(slots: dict[str, str] = {})`; `TemplateOutput(content: str)`; `save_draft(root: Path, spec: DistilledSkillSpec, provenance: DistillProvenance) -> Path` (writes `skills/draft/<name>/` with `template.yaml`, `provenance.yaml`, `SKILL.md`; returns the dir); `load_spec(skill_dir: Path) -> DistilledSkillSpec`.

- [ ] **Step 1: Write the failing test**

Create `src/lottie/distill/tests/test_schema_io.py`:

```python
from pathlib import Path

from lottie.distill.io import load_spec, save_draft
from lottie.distill.schema import DistilledSkillSpec, DistillProvenance


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    spec = DistilledSkillSpec(name="summ", template="Summarize {topic}.", slots=["topic"])
    prov = DistillProvenance(source_agent="Digest", namespace="ns", source_run_ids=["r1", "r2"])
    d = save_draft(tmp_path, spec, prov)

    assert d == tmp_path / "skills" / "draft" / "summ"
    assert (d / "template.yaml").is_file()
    assert (d / "provenance.yaml").is_file()
    assert (d / "SKILL.md").is_file()
    # draft must NOT be a discoverable skill (no skill.py)
    assert not (d / "skill.py").exists()

    loaded = load_spec(d)
    assert loaded.name == "summ"
    assert loaded.template == "Summarize {topic}."
    assert loaded.slots == ["topic"]
    assert loaded.version == "0.1.0"


def test_template_input_output_defaults() -> None:
    from lottie.distill.schema import TemplateInput, TemplateOutput

    assert TemplateInput().slots == {}
    assert TemplateOutput(content="x").content == "x"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/distill/tests/test_schema_io.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.distill'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/lottie/distill/__init__.py`:

```python
from lottie.distill.schema import (
    DistilledSkillSpec,
    DistillProvenance,
    TemplateInput,
    TemplateOutput,
)

__all__ = [
    "DistillProvenance",
    "DistilledSkillSpec",
    "TemplateInput",
    "TemplateOutput",
]
```

Create `src/lottie/distill/schema.py`:

```python
"""Typed contracts for distilled template-skills (V2 S3). Pure data — pydantic only."""

from __future__ import annotations

from pydantic import BaseModel


class DistilledSkillSpec(BaseModel):
    """A reusable prompt template distilled from an agent's learned lessons."""

    name: str
    template: str            # prompt text with {slot} placeholders
    slots: list[str] = []    # slot names (extracted from template, deterministic)
    version: str = "0.1.0"


class DistillProvenance(BaseModel):
    """Where a distilled skill came from (which agent / notes produced it)."""

    source_agent: str
    namespace: str
    source_run_ids: list[str] = []
    version: str = "0.1.0"


class TemplateInput(BaseModel):
    """Input to the generic TemplateRunnerSkill: values for the template's slots."""

    slots: dict[str, str] = {}


class TemplateOutput(BaseModel):
    """Output of TemplateRunnerSkill: the LLM's response to the filled template."""

    content: str
```

Create `src/lottie/distill/io.py`:

```python
"""On-disk persistence for distilled draft skills.

A draft lives in `skills/draft/<name>/` as `template.yaml` + `provenance.yaml` +
`SKILL.md`. It has NO `skill.py`, so `discover_skills` never treats it as callable —
promotion (S3b) is what makes a distilled skill runnable.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from lottie.distill.schema import DistilledSkillSpec, DistillProvenance


def save_draft(root: Path, spec: DistilledSkillSpec, provenance: DistillProvenance) -> Path:
    """Write a distilled draft under `<root>/skills/draft/<spec.name>/`; return the dir."""
    skill_dir = Path(root) / "skills" / "draft" / spec.name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "template.yaml").write_text(
        yaml.safe_dump(spec.model_dump(), sort_keys=True), encoding="utf-8"
    )
    (skill_dir / "provenance.yaml").write_text(
        yaml.safe_dump(provenance.model_dump(), sort_keys=True), encoding="utf-8"
    )
    (skill_dir / "SKILL.md").write_text(
        f"# {spec.name} (distilled draft)\n\n"
        f"Parameterized prompt-template skill, version {spec.version}.\n\n"
        f"**Slots:** {', '.join(spec.slots) or '(none)'}\n\n"
        f"**Provenance:** distilled from agent `{provenance.source_agent}`, "
        f"namespace `{provenance.namespace}`.\n\n"
        "Draft — not callable until a human promotes it (`lottie distill review`).\n",
        encoding="utf-8",
    )
    return skill_dir


def load_spec(skill_dir: Path) -> DistilledSkillSpec:
    """Load the `DistilledSkillSpec` from a draft/promoted skill directory."""
    raw = yaml.safe_load((Path(skill_dir) / "template.yaml").read_text(encoding="utf-8"))
    return DistilledSkillSpec.model_validate(raw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/distill/tests/test_schema_io.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lottie/distill/__init__.py src/lottie/distill/schema.py src/lottie/distill/io.py src/lottie/distill/tests/test_schema_io.py
git commit -m "feat(distill): DistilledSkillSpec + draft I/O (V2 S3a)"
```

---

## Task 2: pure generate helpers

**Files:**
- Create: `src/lottie/distill/generate.py`
- Test: `src/lottie/distill/tests/test_generate.py`

**Interfaces:**
- Consumes: `Message` (`lottie.llm`).
- Produces: `DISTILL_SYSTEM_PROMPT: str`; `build_distill_prompt(notes: list[str]) -> list[Message]`; `extract_slots(template: str) -> list[str]` (sorted unique `{word}` names); `fill_template(template: str, slots: dict[str, str]) -> str` (replaces each `{word}`; raises `KeyError` listing any missing slot).

- [ ] **Step 1: Write the failing test**

Create `src/lottie/distill/tests/test_generate.py`:

```python
import pytest

from lottie.distill.generate import (
    build_distill_prompt,
    extract_slots,
    fill_template,
)


def test_build_prompt_has_system_and_notes() -> None:
    msgs = build_distill_prompt(["use backoff", "cache config"])
    assert [m.role for m in msgs] == ["system", "user"]
    assert "use backoff" in msgs[1].content
    assert "cache config" in msgs[1].content


def test_extract_slots_sorted_unique() -> None:
    assert extract_slots("Summarize {topic} in {n} words about {topic}.") == ["n", "topic"]
    assert extract_slots("no slots here") == []


def test_fill_template_replaces_and_leaves_other_braces() -> None:
    out = fill_template("Return JSON {{ok}} for {topic}", {"topic": "cats"})
    # only {topic} is a slot ({{ok}} has no single-word slot match for 'ok'? see note)
    assert "cats" in out


def test_fill_template_missing_slot_raises() -> None:
    with pytest.raises(KeyError):
        fill_template("Summarize {topic}", {})
```

Note on `test_fill_template_replaces_and_leaves_other_braces`: the slot regex is `{(\w+)}` — it matches `{ok}` inside `{{ok}}` too. If you want literal `{{ }}` to survive, that is a larger escaping design; for S3a keep the simple regex and adjust this test to a template WITHOUT double-brace literals, e.g. `fill_template("Summarize {topic}", {"topic": "cats"})` asserting `== "Summarize cats"`. Prefer the simpler test; do not add brace-escaping in S3a.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/distill/tests/test_generate.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.distill.generate'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/lottie/distill/generate.py`:

```python
"""Pure helpers for distilling lessons into a template (no LLM call here).

The LLM call lives in the CLI so it uses the project's provider. This module builds
the prompt and does deterministic slot extraction/filling.
"""

from __future__ import annotations

import re

from lottie.llm import Message

DISTILL_SYSTEM_PROMPT = (
    "You synthesize ONE reusable prompt template from an agent's learned lessons. "
    "Output ONLY the template text — no commentary, no code fences. Use {slot} "
    "placeholders in snake_case for the parts that vary per invocation."
)

_SLOT_RE = re.compile(r"{(\w+)}")


def build_distill_prompt(notes: list[str]) -> list[Message]:
    """System+user prompt asking the LLM to synthesize a template from `notes`."""
    body = "Learned lessons:\n" + "\n".join(f"- {n}" for n in notes)
    return [
        Message(role="system", content=DISTILL_SYSTEM_PROMPT),
        Message(role="user", content=body),
    ]


def extract_slots(template: str) -> list[str]:
    """Sorted unique `{word}` slot names found in `template`."""
    return sorted(set(_SLOT_RE.findall(template)))


def fill_template(template: str, slots: dict[str, str]) -> str:
    """Replace each `{word}` with `slots[word]`; raise KeyError listing missing slots."""
    missing = [name for name in extract_slots(template) if name not in slots]
    if missing:
        raise KeyError(f"missing slots: {', '.join(missing)}")
    return _SLOT_RE.sub(lambda m: slots[m.group(1)], template)
```

- [ ] **Step 4: Run test to verify it passes** (use the simpler brace test per the note)

Run: `uv run pytest src/lottie/distill/tests/test_generate.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lottie/distill/generate.py src/lottie/distill/tests/test_generate.py
git commit -m "feat(distill): pure generate helpers — prompt, slot extract/fill (V2 S3a)"
```

---

## Task 3: `TemplateRunnerSkill`

**Files:**
- Create: `src/lottie/distill/runner.py`
- Modify: `src/lottie/distill/__init__.py` (export)
- Test: `src/lottie/distill/tests/test_runner.py`

**Interfaces:**
- Consumes: `BaseSkill` (`lottie.core`), `LLMProvider`/`Message` (`lottie.llm`), `DistilledSkillSpec`/`TemplateInput`/`TemplateOutput` (schema), `fill_template` (generate).
- Produces: `TemplateRunnerSkill(BaseSkill[TemplateInput, TemplateOutput])` with `__init__(self, llm: LLMProvider, spec: DistilledSkillSpec, *, name: str | None = None)`; `capability_name = "template_runner"`; `_execute` fills the spec's template from `data.slots` (a missing slot raises `KeyError` → fail-closed) and returns the LLM completion as `TemplateOutput`.

- [ ] **Step 1: Write the failing test**

Create `src/lottie/distill/tests/test_runner.py`:

```python
import pytest

from lottie.distill.runner import TemplateRunnerSkill
from lottie.distill.schema import DistilledSkillSpec, TemplateInput
from lottie.llm import MockLLMProvider


def _spec() -> DistilledSkillSpec:
    return DistilledSkillSpec(name="summ", template="Summarize {topic}.", slots=["topic"])


def test_runner_fills_and_completes() -> None:
    skill = TemplateRunnerSkill(MockLLMProvider(["a summary"]), _spec())
    out = skill.run(TemplateInput(slots={"topic": "otters"}))
    assert out.content == "a summary"


def test_runner_missing_slot_fails_closed() -> None:
    skill = TemplateRunnerSkill(MockLLMProvider(["x"]), _spec())
    with pytest.raises(KeyError):
        skill.run(TemplateInput(slots={}))


def test_runner_sends_filled_prompt() -> None:
    class _Capture(MockLLMProvider):
        def __init__(self) -> None:
            super().__init__(["ok"])
            self.seen = ""

        def complete(self, messages, model_params=None):  # type: ignore[no-untyped-def]
            self.seen = messages[-1].content
            return super().complete(messages, model_params)

    llm = _Capture()
    TemplateRunnerSkill(llm, _spec()).run(TemplateInput(slots={"topic": "otters"}))
    assert llm.seen == "Summarize otters."
```

Note: `test_runner_sends_filled_prompt`'s `_Capture.complete` override uses `# type: ignore[no-untyped-def]` — that is acceptable ONLY in a test and ONLY if a fully-typed override is awkward; PREFER the fully typed signature `def complete(self, messages: list[Message], model_params: Mapping[str, object] | None = None) -> LLMResponse:` (import `Mapping`, `Message`, `LLMResponse`) and drop the ignore. Match the S2a `_CapturingLLM` pattern which did this without an ignore.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/distill/tests/test_runner.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.distill.runner'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/lottie/distill/runner.py`:

```python
"""TemplateRunnerSkill — the single generic executor for distilled template skills.

Fills a DistilledSkillSpec's template from typed slot inputs and calls the LLM. No
LLM-authored code runs (D2): every distilled skill is data (a template) executed here.
"""

from __future__ import annotations

from lottie.core import BaseSkill
from lottie.distill.generate import fill_template
from lottie.distill.schema import DistilledSkillSpec, TemplateInput, TemplateOutput
from lottie.llm import LLMProvider, Message


class TemplateRunnerSkill(BaseSkill[TemplateInput, TemplateOutput]):
    """Execute a distilled template: fill slots, complete, return the text."""

    capability_name = "template_runner"

    def __init__(
        self, llm: LLMProvider, spec: DistilledSkillSpec, *, name: str | None = None
    ) -> None:
        super().__init__(name=name)
        self._llm = llm
        self._spec = spec

    def _execute(self, data: TemplateInput) -> TemplateOutput:
        prompt = fill_template(self._spec.template, data.slots)
        response = self._llm.complete([Message(role="user", content=prompt)])
        return TemplateOutput(content=response.content)
```

Add `TemplateRunnerSkill` to `src/lottie/distill/__init__.py` imports + `__all__` (keep sorted).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/distill/tests/test_runner.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lottie/distill/runner.py src/lottie/distill/__init__.py src/lottie/distill/tests/test_runner.py
git commit -m "feat(distill): TemplateRunnerSkill — generic template executor (V2 S3a)"
```

---

## Task 4: `lottie distill` CLI

**Files:**
- Create: `src/lottie/cli/distill.py`
- Modify: `src/lottie/cli/app.py`
- Test: `src/lottie/cli/tests/test_distill_cli.py`

**Interfaces:**
- Consumes: `find_project_root`/`load_agent_config`, `build_provider`, `build_memory_client`, `MemoryQuery`/`MemoryTier`, `SecretDetectionSkill`, `build_distill_prompt`/`extract_slots`, `DistilledSkillSpec`/`DistillProvenance`, `save_draft`.
- Produces: `distill(name: str, namespace: str | None = None, limit: int = 20, skill_name: str | None = None, provider: str | None = None) -> None` — recalls the agent's SEMANTIC notes, LLM-synthesizes a template, **secret-scans it (reject → exit 2, no draft)**, extracts slots, and writes a draft via `save_draft`. Registered as `lottie distill`. Unknown agent → `typer.BadParameter`. No semantic notes → message + exit 1.

- [ ] **Step 1: Write the failing test**

Create `src/lottie/cli/tests/test_distill_cli.py`:

```python
from pathlib import Path

from typer.testing import CliRunner

from lottie.cli.app import app
from lottie.llm import MockLLMProvider
from lottie.memory.mock import MockMemoryClient
from lottie.memory.schema import MemoryOrigin, MemoryRecord, MemoryTier

runner = CliRunner()


def _project(tmp_path: Path) -> Path:
    (tmp_path / "lottie.yaml").write_text(
        "project: t\nproviders:\n  default: mock\n", encoding="utf-8"
    )
    unit = tmp_path / "agents" / "digest"
    unit.mkdir(parents=True)
    (unit / "agent.py").write_text("# stub\n", encoding="utf-8")
    (unit / "config.yaml").write_text(
        "provider: mock\nmemory:\n  enabled: true\n  backend: mock\n", encoding="utf-8"
    )
    return tmp_path


def _seeded_memory() -> MockMemoryClient:
    mem = MockMemoryClient()
    mem.remember(
        MemoryRecord(
            content="always cite the source", tier=MemoryTier.SEMANTIC, namespace="ns",
            origin=MemoryOrigin.REFLECTION, source_agent="digest", run_id="r1",
        )
    )
    return mem


def test_distill_writes_draft(tmp_path: Path, monkeypatch) -> None:
    root = _project(tmp_path)
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        "lottie.cli.distill.build_provider",
        lambda _m: MockLLMProvider(["Answer about {topic}, always cite the source."]),
    )
    monkeypatch.setattr(
        "lottie.cli.distill.build_memory_client", lambda *_a, **_k: _seeded_memory()
    )
    result = runner.invoke(app, ["distill", "digest", "--namespace", "ns", "--skill-name", "cited"])
    assert result.exit_code == 0, result.stdout
    draft = root / "skills" / "draft" / "cited"
    assert (draft / "template.yaml").is_file()
    assert not (draft / "skill.py").exists()  # draft is not callable


def test_distill_no_notes_exits_1(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(_project(tmp_path))
    monkeypatch.setattr("lottie.cli.distill.build_provider", lambda _m: MockLLMProvider(["x"]))
    monkeypatch.setattr(
        "lottie.cli.distill.build_memory_client", lambda *_a, **_k: MockMemoryClient()
    )
    result = runner.invoke(app, ["distill", "digest", "--namespace", "ns"])
    assert result.exit_code == 1


def test_distill_secret_in_template_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(_project(tmp_path))
    monkeypatch.setattr(
        "lottie.cli.distill.build_provider",
        lambda _m: MockLLMProvider(["use key AKIAIOSFODNN7EXAMPLE for {topic}"]),
    )
    monkeypatch.setattr(
        "lottie.cli.distill.build_memory_client", lambda *_a, **_k: _seeded_memory()
    )
    result = runner.invoke(app, ["distill", "digest", "--namespace", "ns"])
    assert result.exit_code == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/cli/tests/test_distill_cli.py -q`
Expected: FAIL — no `distill` command (`exit_code == 2`, "No such command"). (Once registered, the secret test also expects 2 — distinguish by asserting on `result.stdout` containing the rejection message in Step 4 if needed.)

- [ ] **Step 3: Write minimal implementation**

Create `src/lottie/cli/distill.py`:

```python
"""`lottie distill <agent>` — synthesize a draft template-skill from learned memory.

Recalls the agent's SEMANTIC notes, asks the LLM to synthesize ONE reusable prompt
template, secret-scans it (templates are legitimately instructional, so injection
review is the human promotion gate in S3b), and writes a NON-callable draft under
skills/draft/<name>/. Promotion to a runnable skill is `lottie distill review` (S3b).
"""

from __future__ import annotations

from typing import Annotated

import typer

from lottie.distill.generate import build_distill_prompt, extract_slots
from lottie.distill.io import save_draft
from lottie.distill.schema import DistilledSkillSpec, DistillProvenance
from lottie.llm import build_provider
from lottie.memory.schema import MemoryQuery, MemoryTier
from lottie.memory.store import build_memory_client
from lottie.project.config import find_project_root, load_agent_config
from lottie.security import SecretDetectionSkill


def distill(
    name: str,
    namespace: Annotated[str | None, typer.Option("--namespace")] = None,
    limit: Annotated[int, typer.Option("--limit")] = 20,
    skill_name: Annotated[str | None, typer.Option("--skill-name")] = None,
    provider: Annotated[str | None, typer.Option("--provider")] = None,
) -> None:
    """Distill an agent's learned lessons into a draft template-skill."""
    root = find_project_root()
    unit_dir = root / "agents" / name
    if not (unit_dir / "agent.py").is_file():
        raise typer.BadParameter(f"agent '{name}' not found")

    cfg = load_agent_config(unit_dir)
    llm = build_provider(provider or cfg.provider)
    memory = build_memory_client(root, backend=cfg.memory.backend, path=cfg.memory.path)
    ns = namespace or cfg.memory.namespace or name

    hits = memory.recall(
        MemoryQuery(text="", namespace=ns, tier=MemoryTier.SEMANTIC, limit=limit)
    ).hits
    notes = [h.record.content for h in hits]
    if not notes:
        typer.echo(f"no semantic notes in namespace '{ns}' to distill", err=True)
        raise typer.Exit(code=1)

    template = llm.complete(build_distill_prompt(notes)).content.strip()
    if SecretDetectionSkill().scan_text(template, source="distill"):
        typer.echo("distillation rejected: secret detected in generated template", err=True)
        raise typer.Exit(code=2)

    sname = skill_name or f"{name}_distilled"
    run_ids = sorted({h.record.run_id for h in hits if h.record.run_id})
    spec = DistilledSkillSpec(name=sname, template=template, slots=extract_slots(template))
    provenance = DistillProvenance(source_agent=name, namespace=ns, source_run_ids=run_ids)
    draft_dir = save_draft(root, spec, provenance)

    typer.echo(
        f"distilled draft skill '{sname}' -> {draft_dir} (slots: {spec.slots or 'none'}). "
        "Review and promote to enable it."
    )
```

Edit `src/lottie/cli/app.py` — add `from lottie.cli.distill import distill` and `app.command("distill")(distill)` alongside the other registrations.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/cli/tests/test_distill_cli.py -q`
Expected: PASS (3 tests). Then run `uv run pytest src/lottie/cli -q` to confirm no other CLI test broke from the new command.

- [ ] **Step 5: Commit**

```bash
git add src/lottie/cli/distill.py src/lottie/cli/app.py src/lottie/cli/tests/test_distill_cli.py
git commit -m "feat(cli): lottie distill — draft template-skill from memory (V2 S3a)"
```

---

## Task 5: CLAUDE.md note + full gate

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the note to CLAUDE.md**

Add to the CLI command list (near `lottie reflect`) and a one-line rule near rule 13:

```markdown
lottie distill <agent>                 # synthesize a DRAFT template-skill from an agent's learned memory (HITL-gated; not callable until promoted)
```

Rule note (after rule 13b):

```markdown
13c. **Distilled skills are prompt templates, never generated code** (D2). `lottie distill`
   writes a NON-callable draft to `skills/draft/`; a human promotes it (S3b). The generated
   template is secret-scanned automatically; injection/malice is the human promotion gate.
```

- [ ] **Step 2: Run the full local gate (rule 7b)**

```bash
uv sync --dev --all-extras
uv run ruff check .
uv run mypy --strict src
uv run pytest -q
```
Expected: ruff clean; mypy clean (new `distill` package); pytest all green (1011 + ~15 new S3a tests).

- [ ] **Step 3: Fix any gate failures**

If the CLI test's `distill`/`build_provider` monkeypatch path is wrong, confirm the import is `from lottie.llm import build_provider` inside `cli/distill.py` so `lottie.cli.distill.build_provider` is the patch target. No `Any`, no `# type: ignore` in production.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(distill): note distilled-skills-are-templates + lottie distill (V2 S3a)"
```

---

## Lab round (R25) — separate `lottie-lab` PR, after S3a merges (or fold into S3b's round)

Not part of this plan's commits. Seed an agent's SEMANTIC memory, run `lottie distill`, assert a draft appears under `skills/draft/` with `template.yaml` + provenance and NO `skill.py`; assert a template containing a secret is rejected (exit 2, no draft); build a `TemplateRunnerSkill` from the draft spec and run it against a scripted MockLLM. Mirror the R11/R10 driver harness.

---

## Self-Review

**Spec coverage (epic §3.6 + S3 row of §4, S3a portion):**
- DistilledSkill on-disk format (template + I/O schema names + provenance + version) → Task 1. ✅
- Generic `TemplateRunnerSkill` executes a template, no codegen → Task 3. ✅
- `lottie distill` → draft from the agent's memory notes, provenance-tagged → Task 4. ✅
- Content untrusted → SecretDetection before write (injection deferred to HITL, with rationale) → Task 4 + Global Constraints. ✅
- Draft NOT callable (no skill.py; `discover_skills` skips it) → Task 1 + test. ✅
- Out of scope (promotion, capability, versioning bump, re-distill) → none built (S3b). ✅

**Placeholder scan:** no TBD/TODO; full code in every code step; the brace-escaping edge case and the test `# type: ignore` both carry an explicit "prefer the simpler/typed form" instruction rather than a vague note. ✅

**Type consistency:** `DistilledSkillSpec`/`DistillProvenance`/`TemplateInput`/`TemplateOutput` identical across schema, io, runner, CLI. `build_distill_prompt`/`extract_slots`/`fill_template` identical generate ↔ runner ↔ CLI. `save_draft(root, spec, provenance)`/`load_spec(dir)` identical Task 1 ↔ Task 4. `distill(name, namespace, limit, skill_name, provider)` identical Task 4 def ↔ test. ✅

**Note on scope discipline:** slots are extracted deterministically from the template (regex), so the LLM never has to reliably emit a slot list — it just emits the template. The secret-scan-not-injection-scan choice is deliberate (a template is instructional by nature); the human promotion gate (S3b) is the injection/malice boundary. Drafts are inert files with no `skill.py`, so nothing in the existing registry can execute them pre-promotion.
