# Phase 0 Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the five open Phase 0 Deliverables Checklist items — runnable hello-world on `init`, CI, `LOTTIE.md` sync on `status`, the new `SchemaValidatorSkill` + security skills, and the AI-powered `ScaffolderAgent` (`--from-desc`).

**Architecture:** A new self-contained `lottie.security` package holds three deterministic scan skills (`SecretDetectionSkill`, `CodeSecurityScanSkill`, `SchemaValidatorSkill`) plus a `guard_and_write` gate that runs them in rule-13 order. `ScaffolderAgent` (in `lottie.scaffold`) calls the LLM for a structured `ScaffoldPlan`, renders new dynamic Jinja templates, and writes through the gate with one retry. Three independent smaller changes (`status`→`LOTTIE.md` sync, bundled hello agent, CI workflow) round out the phase.

**Tech Stack:** Python 3.12, Pydantic v2, Typer, Jinja2, litellm (via `LLMProvider`), `detect-secrets`, `bandit`, `mypy --strict`, `ruff`, pytest. Package + tooling managed with `uv` (run all tooling from the project dir).

---

## Conventions for every task

- All commands run from the project root `/Users/cdiaz19/Documents/trae_projects/lottie-orchestrator` unless stated.
- Run tooling via `uv run <tool>` so it uses the project venv.
- Built-in framework skills are documented with a thorough module docstring (the `TemplateRendererSkill` precedent), **not** a `SKILL.md` file. `SKILL.md`/`AGENT.md` files are only produced for scaffolded *user* units.
- Tests never call a real LLM — use `MockLLMProvider` (CLAUDE.md rule 5).
- Commit after each task with a conventional-commit message.

## Architecture notes / known risks

- **Gate write-then-rollback:** `mypy --strict` cannot resolve a generated agent's relative imports (`from .schema import ...`) from a temp dir, so `guard_and_write` writes into the real target dir, runs scanners with `cwd=project root`, and `shutil.rmtree`s the target on failure. Net effect = zero surviving files on failure. This refines the spec's "render-to-temp" wording while preserving its "no partial writes" intent.
- **`SchemaValidatorSkill` placement:** lives in `security/` (not `scaffold/`) to keep `security` free of any `scaffold` import (avoids an import cycle, since `scaffold` imports `security`).
- **LLM `run_body` fragility:** mypy `--strict` + ruff over LLM-written `run()` bodies can fail (unused imports, type errors). The 1-retry loop feeds diagnostics back; persistent failure hard-fails with a clear message. This is by design, not a bug.

---

## Task 1: Security package — dependencies and shared schema

**Files:**
- Modify: `pyproject.toml`
- Create: `src/lottie/security/schema.py`
- Create: `src/lottie/security/tests/__init__.py`
- Test: `src/lottie/security/tests/test_schema.py`

- [ ] **Step 1: Add runtime deps and mypy overrides**

Run: `uv add detect-secrets bandit`

Then add to `pyproject.toml` (after the existing `[tool.mypy]` block) so strict mode tolerates the untyped third-party libs:

```toml
[[tool.mypy.overrides]]
module = ["detect_secrets.*", "bandit.*"]
ignore_missing_imports = true
```

- [ ] **Step 2: Write the failing test for the shared schema**

Create `src/lottie/security/tests/__init__.py` (empty file).

Create `src/lottie/security/tests/test_schema.py`:

```python
from __future__ import annotations

from lottie.security.schema import (
    GateResult,
    ScanInput,
    ScanOutput,
    SecurityFinding,
    ValidateInput,
    ValidateOutput,
)


def test_scan_output_defaults_empty() -> None:
    assert ScanOutput().findings == []


def test_scan_input_holds_paths() -> None:
    assert ScanInput(paths=["a.py", "b.py"]).paths == ["a.py", "b.py"]


def test_security_finding_fields() -> None:
    f = SecurityFinding(file="x.py", line=3, kind="AWSKeyDetector", message="secret")
    assert f.file == "x.py"
    assert f.line == 3


def test_gate_result_passed_flag() -> None:
    r = GateResult(passed=True, findings=[], diagnostics="", files_written=["x.py"])
    assert r.passed is True
    assert r.files_written == ["x.py"]


def test_validate_output_shape() -> None:
    v = ValidateOutput(passed=False, diagnostics="error: bad")
    assert v.passed is False
    assert "bad" in v.diagnostics
    assert ValidateInput(paths=["x.py"]).paths == ["x.py"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest src/lottie/security/tests/test_schema.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.security.schema'`

- [ ] **Step 4: Write the schema**

Create `src/lottie/security/schema.py`:

```python
"""Typed contracts for the security scan skills and the code-write gate.

`ScanInput`/`ScanOutput` are shared by the content scanners (secret + bandit).
`ValidateInput`/`ValidateOutput` carry the mypy+ruff verdict. `GateResult` is the
combined outcome returned by `guard_and_write`.
"""

from __future__ import annotations

from pydantic import BaseModel


class SecurityFinding(BaseModel):
    """A single issue located in a scanned file."""

    file: str
    line: int
    kind: str
    message: str


class ScanInput(BaseModel):
    """File paths to scan."""

    paths: list[str]


class ScanOutput(BaseModel):
    """Findings from a content scanner."""

    findings: list[SecurityFinding] = []


class ValidateInput(BaseModel):
    """File paths to type-check and lint."""

    paths: list[str]


class ValidateOutput(BaseModel):
    """Combined mypy + ruff verdict."""

    passed: bool
    diagnostics: str = ""


class GateResult(BaseModel):
    """Outcome of the rule-13 code-write gate."""

    passed: bool
    findings: list[SecurityFinding] = []
    diagnostics: str = ""
    files_written: list[str] = []
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest src/lottie/security/tests/test_schema.py -q`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/lottie/security/schema.py src/lottie/security/tests/
git commit -m "feat(security): add scan/validate/gate schemas and deps"
```

---

## Task 2: SecretDetectionSkill

**Files:**
- Create: `src/lottie/security/secret_detector.py`
- Test: `src/lottie/security/tests/test_secret_detector.py`

- [ ] **Step 1: Write the failing test**

Create `src/lottie/security/tests/test_secret_detector.py`:

```python
from __future__ import annotations

from pathlib import Path

from lottie.security.schema import ScanInput
from lottie.security.secret_detector import SecretDetectionSkill


def _write(tmp_path: Path, name: str, content: str) -> str:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


def test_detects_aws_access_key(tmp_path: Path) -> None:
    path = _write(tmp_path, "leak.py", 'KEY = "AKIAIOSFODNN7EXAMPLE"\n')
    out = SecretDetectionSkill(enable_benchmarks=False).run(ScanInput(paths=[path]))
    assert any(f.kind == "AWSAccessKey" for f in out.findings)


def test_detects_private_key_header(tmp_path: Path) -> None:
    path = _write(tmp_path, "id_rsa", "-----BEGIN RSA PRIVATE KEY-----\n")
    out = SecretDetectionSkill(enable_benchmarks=False).run(ScanInput(paths=[path]))
    assert any(f.kind == "PrivateKey" for f in out.findings)


def test_clean_file_has_no_findings(tmp_path: Path) -> None:
    path = _write(tmp_path, "ok.py", "x = 1 + 2\n")
    out = SecretDetectionSkill(enable_benchmarks=False).run(ScanInput(paths=[path]))
    assert out.findings == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/security/tests/test_secret_detector.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.security.secret_detector'`

- [ ] **Step 3: Write the skill**

Create `src/lottie/security/secret_detector.py`:

```python
"""SecretDetectionSkill — flag secrets in files before they are written or shipped.

Runs the `detect-secrets` plugin suite over each file and augments it with a small
set of high-signal custom regexes. Deterministic: same file content always yields
the same findings. Applied at knowledge ingest, on LLM outputs, and inside the
code-write gate (CLAUDE.md rules 9, 10, 13).
"""

from __future__ import annotations

import re
from pathlib import Path

from detect_secrets import SecretsCollection
from detect_secrets.settings import default_settings

from lottie.core import BaseSkill
from lottie.security.schema import ScanInput, ScanOutput, SecurityFinding

_CUSTOM_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("PrivateKey", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("AWSAccessKey", re.compile(r"AKIA[0-9A-Z]{16}")),
]


class SecretDetectionSkill(BaseSkill[ScanInput, ScanOutput]):
    """Scan files for secrets via detect-secrets plus custom patterns."""

    def _execute(self, data: ScanInput) -> ScanOutput:
        seen: set[tuple[str, int, str]] = set()
        findings: list[SecurityFinding] = []
        for raw in data.paths:
            path = Path(raw)
            if not path.is_file():
                continue
            self._scan_detect_secrets(raw, seen, findings)
            self._scan_custom(raw, path, seen, findings)
        return ScanOutput(findings=findings)

    def _scan_detect_secrets(
        self,
        raw: str,
        seen: set[tuple[str, int, str]],
        findings: list[SecurityFinding],
    ) -> None:
        collection = SecretsCollection()
        with default_settings():
            collection.scan_file(raw)
        for _filename, secret in collection:
            key = (raw, secret.line_number, secret.type)
            if key in seen:
                continue
            seen.add(key)
            findings.append(
                SecurityFinding(
                    file=raw,
                    line=secret.line_number,
                    kind=secret.type,
                    message="potential secret",
                )
            )

    def _scan_custom(
        self,
        raw: str,
        path: Path,
        seen: set[tuple[str, int, str]],
        findings: list[SecurityFinding],
    ) -> None:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for kind, pattern in _CUSTOM_PATTERNS:
                if pattern.search(line):
                    key = (raw, lineno, kind)
                    if key in seen:
                        continue
                    seen.add(key)
                    findings.append(
                        SecurityFinding(
                            file=raw, line=lineno, kind=kind, message="potential secret"
                        )
                    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/security/tests/test_secret_detector.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy --strict src/lottie/security/secret_detector.py && uv run ruff check src/lottie/security/secret_detector.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/lottie/security/secret_detector.py src/lottie/security/tests/test_secret_detector.py
git commit -m "feat(security): add SecretDetectionSkill"
```

---

## Task 3: CodeSecurityScanSkill (bandit)

**Files:**
- Create: `src/lottie/security/code_scanner.py`
- Test: `src/lottie/security/tests/test_code_scanner.py`

- [ ] **Step 1: Write the failing test**

Create `src/lottie/security/tests/test_code_scanner.py`:

```python
from __future__ import annotations

from pathlib import Path

from lottie.security.code_scanner import CodeSecurityScanSkill
from lottie.security.schema import ScanInput


def _write(tmp_path: Path, name: str, content: str) -> str:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


def test_flags_shell_injection(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "danger.py",
        "import subprocess\n\n\ndef go(cmd: str) -> None:\n    subprocess.call(cmd, shell=True)\n",
    )
    out = CodeSecurityScanSkill(enable_benchmarks=False).run(ScanInput(paths=[path]))
    assert out.findings, "expected at least one bandit finding"
    assert all(f.file == path for f in out.findings)


def test_clean_file_has_no_findings(tmp_path: Path) -> None:
    path = _write(tmp_path, "ok.py", "def add(a: int, b: int) -> int:\n    return a + b\n")
    out = CodeSecurityScanSkill(enable_benchmarks=False).run(ScanInput(paths=[path]))
    assert out.findings == []


def test_empty_paths_returns_empty() -> None:
    out = CodeSecurityScanSkill(enable_benchmarks=False).run(ScanInput(paths=[]))
    assert out.findings == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/security/tests/test_code_scanner.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.security.code_scanner'`

- [ ] **Step 3: Write the skill**

Create `src/lottie/security/code_scanner.py`:

```python
"""CodeSecurityScanSkill — run bandit over generated/modified Python.

Step 3 of the code review pipeline (CLAUDE.md rule 13). Shells out to bandit in
JSON mode and normalizes its results into `SecurityFinding`s. semgrep is deferred
to a later phase. Deterministic given the same source.
"""

from __future__ import annotations

import json
import subprocess
import sys

from lottie.core import BaseSkill
from lottie.security.schema import ScanInput, ScanOutput, SecurityFinding


class CodeSecurityScanSkill(BaseSkill[ScanInput, ScanOutput]):
    """Scan Python files with bandit; return one finding per reported issue."""

    def _execute(self, data: ScanInput) -> ScanOutput:
        if not data.paths:
            return ScanOutput(findings=[])
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", "bandit", "-f", "json", "-q", *data.paths],
            capture_output=True,
            text=True,
        )
        try:
            report = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return ScanOutput(findings=[])
        findings = [
            SecurityFinding(
                file=str(result["filename"]),
                line=int(result["line_number"]),
                kind=str(result["test_id"]),
                message=str(result["issue_text"]),
            )
            for result in report.get("results", [])
        ]
        return ScanOutput(findings=findings)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/security/tests/test_code_scanner.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy --strict src/lottie/security/code_scanner.py && uv run ruff check src/lottie/security/code_scanner.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/lottie/security/code_scanner.py src/lottie/security/tests/test_code_scanner.py
git commit -m "feat(security): add CodeSecurityScanSkill (bandit)"
```

---

## Task 4: SchemaValidatorSkill (mypy + ruff)

**Files:**
- Create: `src/lottie/security/validator.py`
- Test: `src/lottie/security/tests/test_validator.py`

- [ ] **Step 1: Write the failing test**

Create `src/lottie/security/tests/test_validator.py`:

```python
from __future__ import annotations

from pathlib import Path

from lottie.security.schema import ValidateInput
from lottie.security.validator import SchemaValidatorSkill


def _write(tmp_path: Path, name: str, content: str) -> str:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


def test_clean_file_passes(tmp_path: Path) -> None:
    path = _write(tmp_path, "ok.py", "def add(a: int, b: int) -> int:\n    return a + b\n")
    out = SchemaValidatorSkill(enable_benchmarks=False).run(ValidateInput(paths=[path]))
    assert out.passed is True


def test_type_error_fails(tmp_path: Path) -> None:
    path = _write(tmp_path, "bad.py", "x: int = 'not an int'\n")
    out = SchemaValidatorSkill(enable_benchmarks=False).run(ValidateInput(paths=[path]))
    assert out.passed is False
    assert out.diagnostics != ""


def test_lint_error_fails(tmp_path: Path) -> None:
    path = _write(tmp_path, "unused.py", "import os\n")
    out = SchemaValidatorSkill(enable_benchmarks=False).run(ValidateInput(paths=[path]))
    assert out.passed is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/security/tests/test_validator.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.security.validator'`

- [ ] **Step 3: Write the skill**

Create `src/lottie/security/validator.py`:

```python
"""SchemaValidatorSkill — type-check and lint generated files (rule-13 steps 4-5).

Runs `mypy --strict` then `ruff check` over the given paths as subprocesses and
reports a single pass/fail plus the combined diagnostics. Used by the code-write
gate; callers pass project-relative paths and run from the project root so mypy
resolves package and relative imports.
"""

from __future__ import annotations

import subprocess
import sys

from lottie.core import BaseSkill
from lottie.security.schema import ValidateInput, ValidateOutput


class SchemaValidatorSkill(BaseSkill[ValidateInput, ValidateOutput]):
    """Validate files with mypy --strict and ruff check."""

    def _execute(self, data: ValidateInput) -> ValidateOutput:
        if not data.paths:
            return ValidateOutput(passed=True, diagnostics="")
        mypy = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", "mypy", "--strict", *data.paths],
            capture_output=True,
            text=True,
        )
        ruff = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", "ruff", "check", *data.paths],
            capture_output=True,
            text=True,
        )
        passed = mypy.returncode == 0 and ruff.returncode == 0
        diagnostics = "".join(
            [mypy.stdout, mypy.stderr, ruff.stdout, ruff.stderr]
        ).strip()
        return ValidateOutput(passed=passed, diagnostics=diagnostics)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/security/tests/test_validator.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy --strict src/lottie/security/validator.py && uv run ruff check src/lottie/security/validator.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/lottie/security/validator.py src/lottie/security/tests/test_validator.py
git commit -m "feat(security): add SchemaValidatorSkill (mypy + ruff)"
```

---

## Task 5: Code-write gate (guard_and_write) + package exports

**Files:**
- Create: `src/lottie/security/write_gate.py`
- Modify: `src/lottie/security/__init__.py`
- Test: `src/lottie/security/tests/test_write_gate.py`

- [ ] **Step 1: Write the failing test**

Create `src/lottie/security/tests/test_write_gate.py`:

```python
from __future__ import annotations

from pathlib import Path

from lottie.security.write_gate import guard_and_write

_CLEAN = {
    "__init__.py": "",
    "mod.py": "def add(a: int, b: int) -> int:\n    return a + b\n",
}


def test_clean_files_are_written(tmp_path: Path) -> None:
    target = tmp_path / "unit"
    result = guard_and_write(target, _CLEAN, root=tmp_path)
    assert result.passed is True
    assert (target / "mod.py").is_file()
    assert any(p.endswith("mod.py") for p in result.files_written)


def test_secret_aborts_and_rolls_back(tmp_path: Path) -> None:
    target = tmp_path / "unit"
    files = {"mod.py": 'KEY = "AKIAIOSFODNN7EXAMPLE"\n'}
    result = guard_and_write(target, files, root=tmp_path)
    assert result.passed is False
    assert result.findings, "expected a secret finding"
    assert not target.exists(), "target dir must be removed on failure"


def test_type_error_aborts_and_rolls_back(tmp_path: Path) -> None:
    target = tmp_path / "unit"
    files = {"mod.py": "x: int = 'bad'\n"}
    result = guard_and_write(target, files, root=tmp_path)
    assert result.passed is False
    assert result.diagnostics != ""
    assert not target.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/security/tests/test_write_gate.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.security.write_gate'`

- [ ] **Step 3: Write the gate**

Create `src/lottie/security/write_gate.py`:

```python
"""guard_and_write — the rule-13 code-write gate.

Writes rendered files into the target directory, then runs the pipeline in order
(SecretDetection -> CodeSecurityScan -> mypy -> ruff). On any failure the target
directory is removed so no partial or unsafe code survives. Scanners run with the
project root as cwd so mypy resolves package and relative imports.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from lottie.security.code_scanner import CodeSecurityScanSkill
from lottie.security.schema import GateResult, ScanInput, ValidateInput
from lottie.security.secret_detector import SecretDetectionSkill
from lottie.security.validator import SchemaValidatorSkill


def guard_and_write(target: Path, files: dict[str, str], root: Path) -> GateResult:
    """Write `files` under `target`, gate them, and roll back on failure.

    `files` maps a path relative to `target` to its content. `root` is the project
    root used to derive scanner-friendly relative paths.
    """
    written: list[Path] = []
    for relpath, content in files.items():
        path = target / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written.append(path)

    all_paths = [_rel(p, root) for p in written]
    py_paths = [
        _rel(p, root) for p in written if p.suffix == ".py" and p.stat().st_size > 0
    ]

    secrets = SecretDetectionSkill(enable_benchmarks=False).run(ScanInput(paths=all_paths))
    bandit = CodeSecurityScanSkill(enable_benchmarks=False).run(ScanInput(paths=py_paths))
    validation = SchemaValidatorSkill(enable_benchmarks=False).run(
        ValidateInput(paths=py_paths)
    )

    findings = [*secrets.findings, *bandit.findings]
    passed = not findings and validation.passed
    if not passed:
        shutil.rmtree(target, ignore_errors=True)
        return GateResult(
            passed=False,
            findings=findings,
            diagnostics=validation.diagnostics,
            files_written=[],
        )
    return GateResult(
        passed=True,
        findings=[],
        diagnostics=validation.diagnostics,
        files_written=all_paths,
    )


def _rel(path: Path, root: Path) -> str:
    """Path relative to root when possible, else absolute — as a string for skills."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
```

- [ ] **Step 4: Write package exports**

Replace `src/lottie/security/__init__.py` with:

```python
"""Security layer — input/output gates and the code-write pipeline."""

from lottie.security.code_scanner import CodeSecurityScanSkill
from lottie.security.schema import (
    GateResult,
    ScanInput,
    ScanOutput,
    SecurityFinding,
    ValidateInput,
    ValidateOutput,
)
from lottie.security.secret_detector import SecretDetectionSkill
from lottie.security.validator import SchemaValidatorSkill
from lottie.security.write_gate import guard_and_write

__all__ = [
    "CodeSecurityScanSkill",
    "GateResult",
    "ScanInput",
    "ScanOutput",
    "SchemaValidatorSkill",
    "SecretDetectionSkill",
    "SecurityFinding",
    "ValidateInput",
    "ValidateOutput",
    "guard_and_write",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest src/lottie/security/tests/test_write_gate.py -q`
Expected: PASS (3 passed)

- [ ] **Step 6: Typecheck, lint, full security suite**

Run: `uv run mypy --strict src/lottie/security && uv run ruff check src/lottie/security && uv run pytest src/lottie/security -q`
Expected: no errors; all security tests pass

- [ ] **Step 7: Commit**

```bash
git add src/lottie/security/
git commit -m "feat(security): add guard_and_write code-write gate"
```

---

## Task 6: `lottie status` regenerates LOTTIE.md (deliverable #5)

**Files:**
- Create: `src/lottie/project/lottie_md.py`
- Modify: `src/lottie/cli/status.py`
- Test: `src/lottie/project/tests/test_lottie_md.py`

- [ ] **Step 1: Write the failing test**

Create `src/lottie/project/tests/test_lottie_md.py`:

```python
from __future__ import annotations

from pathlib import Path

from lottie.project.lottie_md import sync

_TEMPLATE = """# demo

## Agents
_None yet._

## Skills
_None yet._
"""


def _project(tmp_path: Path) -> Path:
    (tmp_path / "lottie.yaml").write_text("project: demo\n", encoding="utf-8")
    (tmp_path / "LOTTIE.md").write_text(_TEMPLATE, encoding="utf-8")
    agent_dir = tmp_path / "agents" / "researcher"
    agent_dir.mkdir(parents=True)
    (agent_dir / "agent.py").write_text("# agent\n", encoding="utf-8")
    skill_dir = tmp_path / "skills" / "web_search"
    skill_dir.mkdir(parents=True)
    (skill_dir / "skill.py").write_text("# skill\n", encoding="utf-8")
    return tmp_path


def test_sync_writes_entries(tmp_path: Path) -> None:
    root = _project(tmp_path)
    sync(root)
    md = (root / "LOTTIE.md").read_text()
    assert "- **ResearcherAgent** — `agents/researcher/`" in md
    assert "- **WebSearchSkill** — `skills/web_search/`" in md
    assert "_None yet_" not in md


def test_sync_is_idempotent(tmp_path: Path) -> None:
    root = _project(tmp_path)
    sync(root)
    first = (root / "LOTTIE.md").read_text()
    sync(root)
    assert (root / "LOTTIE.md").read_text() == first


def test_sync_restores_placeholder_when_empty(tmp_path: Path) -> None:
    (tmp_path / "lottie.yaml").write_text("project: demo\n", encoding="utf-8")
    (tmp_path / "LOTTIE.md").write_text(_TEMPLATE, encoding="utf-8")
    sync(tmp_path)
    md = (tmp_path / "LOTTIE.md").read_text()
    assert md.count("_None yet_") == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/project/tests/test_lottie_md.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.project.lottie_md'`

- [ ] **Step 3: Write the module**

Create `src/lottie/project/lottie_md.py`:

```python
"""Regenerate the LOTTIE.md registry sections from on-disk discovery.

`sync` rebuilds the `## Agents` and `## Skills` sections wholesale from what is
actually present, so `lottie status` keeps LOTTIE.md current. Filesystem-only
(via discovery) — never imports user code — and idempotent.
"""

from __future__ import annotations

from pathlib import Path

from lottie.project.discovery import UnitInfo, discover_agents, discover_skills
from lottie.scaffold.generator import _class_name

_PLACEHOLDER = "_None yet_"
_AGENTS_HEADING = "## Agents"
_SKILLS_HEADING = "## Skills"


def sync(root: Path) -> None:
    """Rewrite LOTTIE.md's Agents/Skills sections from discovery. No-op if absent."""
    md = root / "LOTTIE.md"
    if not md.exists():
        return
    text = md.read_text(encoding="utf-8")
    text = _replace_section(text, _AGENTS_HEADING, _entries(discover_agents(root), "agent"))
    text = _replace_section(text, _SKILLS_HEADING, _entries(discover_skills(root), "skill"))
    md.write_text(text, encoding="utf-8")


def _entries(units: list[UnitInfo], kind: str) -> list[str]:
    if not units:
        return [_PLACEHOLDER]
    return [
        f"- **{_class_name(u.name, kind)}** — `{kind}s/{u.name}/`"  # type: ignore[arg-type]
        for u in units
    ]


def _replace_section(text: str, heading: str, body: list[str]) -> str:
    """Replace lines from `heading` until the next `## ` (or EOF) with heading+body."""
    lines = text.splitlines()
    try:
        start = lines.index(heading)
    except ValueError:
        return text
    end = start + 1
    while end < len(lines) and not lines[end].startswith("## "):
        end += 1
    rebuilt = [heading, "", *body, ""]
    new_lines = [*lines[:start], *rebuilt, *lines[end:]]
    return "\n".join(new_lines).rstrip("\n") + "\n"
```

Note: `_class_name`'s parameter is typed `Literal["agent", "skill"]`; the `kind: str` here is narrowed by callers passing literals, hence the inline `type: ignore[arg-type]`.

- [ ] **Step 4: Wire it into status**

In `src/lottie/cli/status.py`, add the import and call `sync` before printing. Change the import block and the start of `status()`:

```python
from lottie.project.config import find_project_root, load_lottie_config
from lottie.project.discovery import UnitInfo, discover_agents, discover_skills
from lottie.project.lottie_md import sync
```

```python
def status() -> None:
    """Show registered agents, skills, knowledge size, and provider config."""
    root = find_project_root()
    sync(root)
    cfg = load_lottie_config(root)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest src/lottie/project/tests/test_lottie_md.py src/lottie/cli/tests/test_status.py -q`
Expected: PASS (all green)

- [ ] **Step 6: Typecheck and lint**

Run: `uv run mypy --strict src/lottie/project/lottie_md.py src/lottie/cli/status.py && uv run ruff check src/lottie/project/lottie_md.py src/lottie/cli/status.py`
Expected: no errors

- [ ] **Step 7: Commit**

```bash
git add src/lottie/project/lottie_md.py src/lottie/cli/status.py src/lottie/project/tests/test_lottie_md.py
git commit -m "feat(cli): lottie status regenerates LOTTIE.md registry"
```

---

## Task 7: Runnable hello-world agent on `lottie init` (deliverable #1)

**Files:**
- Create: `src/lottie/scaffold/hello.py`
- Modify: `src/lottie/cli/init.py`
- Test: `src/lottie/cli/tests/test_init.py` (add cases)

- [ ] **Step 1: Write the failing test (append to existing file)**

Add to `src/lottie/cli/tests/test_init.py`:

```python
import subprocess
import sys
from types import SimpleNamespace


def test_init_creates_runnable_hello_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    base = tmp_path / "demo" / "agents" / "hello"
    for rel in ["agent.py", "schema.py", "config.yaml", "prompts.py", "AGENT.md",
                "tests/test_hello.py"]:
        assert (base / rel).is_file(), f"missing {rel}"
    assert "class HelloAgent(BaseAgent" in (base / "agent.py").read_text()


def test_init_hello_tests_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(demo / "agents" / "hello")],
        cwd=demo,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_init_hello_runs_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    monkeypatch.chdir(tmp_path / "demo")

    def fake_completion(model: str, messages: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Hello, Ada!"))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
        )

    monkeypatch.setattr("litellm.completion", fake_completion)
    result = runner.invoke(app, ["run", "hello", "--input", '{"name": "Ada"}'])
    assert result.exit_code == 0, result.output
    assert "Hello, Ada!" in result.output
```

Confirm the top of `test_init.py` already imports `Path`, `pytest`, `runner`, and `app` (it does — it uses them in existing tests). If `Path`/`pytest` are missing, add `from pathlib import Path` and `import pytest`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/cli/tests/test_init.py -q -k hello`
Expected: FAIL — hello agent files not created

- [ ] **Step 3: Write the bundled hello agent**

Create `src/lottie/scaffold/hello.py`:

```python
"""Bundled, runnable starter agent written by `lottie init`.

HELLO_FILES maps a path (relative to agents/hello/) to its literal content. The
agent calls the configured LLM provider to greet the user, so `lottie run hello`
works end-to-end immediately and its MockLLM tests pass out of the box.
"""

from __future__ import annotations

_AGENT_PY = '''\
"""HelloAgent — a runnable starter agent created by `lottie init`."""
from __future__ import annotations
from lottie.core import BaseAgent
from lottie.llm import Message
from .prompts import SYSTEM_PROMPT
from .schema import HelloInput, HelloOutput


class HelloAgent(BaseAgent[HelloInput, HelloOutput]):
    """Greets the user via the configured LLM provider."""

    def _execute(self, data: HelloInput) -> HelloOutput:
        response = self.complete(
            [
                Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content=f"Greet {data.name}."),
            ]
        )
        return HelloOutput(greeting=response.content)
'''

_SCHEMA_PY = '''\
"""Typed input/output models for HelloAgent."""
from __future__ import annotations
from pydantic import BaseModel


class HelloInput(BaseModel):
    """Input for HelloAgent."""
    name: str = "world"


class HelloOutput(BaseModel):
    """Output from HelloAgent."""
    greeting: str
'''

_PROMPTS_PY = '''\
"""Prompt templates for HelloAgent."""
from __future__ import annotations

SYSTEM_PROMPT = """\\
You are HelloAgent, a friendly Lottie starter agent.
Greet the user warmly in one short sentence.
"""
'''

_CONFIG_YAML = """\
provider: anthropic/claude-sonnet-4-6
model_params:
  temperature: 0.3
  max_tokens: 256
capabilities: []
policies:
  - base
memory:
  enabled: false
  namespace: hello
"""

_AGENT_MD = """\
# HelloAgent

## Role
Greets the user — the runnable starter agent created by `lottie init`.

## Input
| Field | Type | Description |
|---|---|---|
| name | str | Who to greet (defaults to "world") |

## Output
| Field | Type | Description |
|---|---|---|
| greeting | str | The generated greeting |

## Provider
Default: anthropic/claude-sonnet-4-6

## Tools (Skills used)
_None yet._

## Policies
- base

## Examples
### Example 1
Input: `{"name": "Ada"}`
Output: `{"greeting": "Hello, Ada!"}`
"""

_TEST_PY = '''\
"""Integration tests for HelloAgent (MockLLMProvider — no real LLM)."""
from __future__ import annotations
from lottie.llm import MockLLMProvider
from agents.hello.agent import HelloAgent
from agents.hello.schema import HelloInput, HelloOutput


def test_hello_greets_via_llm() -> None:
    agent = HelloAgent(llm=MockLLMProvider(["Hello, Ada!"]))
    result = agent.run(HelloInput(name="Ada"))
    assert isinstance(result, HelloOutput)
    assert result.greeting == "Hello, Ada!"


def test_hello_makes_one_llm_call() -> None:
    mock = MockLLMProvider(["hi"])
    HelloAgent(llm=mock).run(HelloInput(name="Ada"))
    assert len(mock.calls) == 1


def test_hello_defaults_to_world() -> None:
    agent = HelloAgent(llm=MockLLMProvider(["Hello, world!"]))
    result = agent.run(HelloInput())
    assert result.greeting == "Hello, world!"
'''

HELLO_FILES: dict[str, str] = {
    "__init__.py": "",
    "AGENT.md": _AGENT_MD,
    "agent.py": _AGENT_PY,
    "schema.py": _SCHEMA_PY,
    "config.yaml": _CONFIG_YAML,
    "prompts.py": _PROMPTS_PY,
    "tests/__init__.py": "",
    "tests/test_hello.py": _TEST_PY,
}
```

- [ ] **Step 4: Wire hello into init**

In `src/lottie/cli/init.py`, add imports and write the hello agent at the end of `_scaffold`, then sync LOTTIE.md.

Add to the import block:

```python
from lottie.cli import templates
from lottie.project.lottie_md import sync
from lottie.scaffold.hello import HELLO_FILES
```

At the end of `_scaffold(target, name)`, after the knowledge-layer loop, append:

```python
    hello_dir = target / "agents" / "hello"
    for relpath, content in HELLO_FILES.items():
        path = hello_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    sync(target)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest src/lottie/cli/tests/test_init.py -q`
Expected: PASS (existing + 3 new hello tests)

- [ ] **Step 6: Typecheck and lint**

Run: `uv run mypy --strict src/lottie/scaffold/hello.py src/lottie/cli/init.py && uv run ruff check src/lottie/scaffold/hello.py src/lottie/cli/init.py`
Expected: no errors

- [ ] **Step 7: Commit**

```bash
git add src/lottie/scaffold/hello.py src/lottie/cli/init.py src/lottie/cli/tests/test_init.py
git commit -m "feat(cli): lottie init scaffolds a runnable hello-world agent"
```

---

## Task 8: CI/CD workflow (deliverable #2)

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write the workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.12"
      - name: Sync dependencies
        run: uv sync --dev
      - name: Lint
        run: uv run ruff check .
      - name: Type check
        run: uv run mypy --strict src
      - name: Tests with coverage (report only)
        run: uv run pytest -q --cov=lottie --cov-report=term-missing
```

- [ ] **Step 2: Validate YAML locally**

Run: `uv run python -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('.github/workflows/ci.yml').read_text()); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Verify the three gates pass locally (what CI will run)**

Run: `uv run ruff check . && uv run mypy --strict src && uv run pytest -q`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add ruff/mypy/pytest workflow on push and PR to main"
```

---

## Task 9: ScaffolderAgent schemas + generalized render context

**Files:**
- Modify: `src/lottie/scaffold/schema.py`
- Test: `src/lottie/scaffold/tests/test_plan_schema.py`

- [ ] **Step 1: Write the failing test**

Create `src/lottie/scaffold/tests/test_plan_schema.py`:

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError

from lottie.scaffold.schema import FieldSpec, PlanRenderContext, ScaffoldPlan, ScaffoldRequest


def test_field_spec_defaults() -> None:
    f = FieldSpec(name="query", type="str")
    assert f.description == ""


def test_scaffold_plan_requires_fields() -> None:
    with pytest.raises(ValidationError):
        ScaffoldPlan(
            class_name="FooAgent",
            input_fields=[],
            output_fields=[FieldSpec(name="result", type="str")],
            system_prompt="hi",
            run_body="return FooAgentOutput(result='x')",
        )


def test_scaffold_plan_valid() -> None:
    plan = ScaffoldPlan(
        class_name="FooAgent",
        input_fields=[FieldSpec(name="query", type="str")],
        output_fields=[FieldSpec(name="result", type="str")],
        system_prompt="hi",
        run_body="return FooAgentOutput(result='x')",
    )
    assert plan.tools == []


def test_scaffold_request_fields() -> None:
    req = ScaffoldRequest(kind="agent", name="foo", description="does foo")
    assert req.repair_feedback is None


def test_plan_render_context_accepts_field_dicts() -> None:
    ctx = PlanRenderContext(
        name="foo",
        class_name="FooAgent",
        provider="anthropic/claude-sonnet-4-6",
        kind="agent",
        input_fields=[FieldSpec(name="query", type="str")],
        output_fields=[FieldSpec(name="result", type="str")],
        system_prompt="hi",
        run_body="...",
        tools=[],
        input_sample='query="x"',
    )
    assert ctx.input_sample == 'query="x"'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/scaffold/tests/test_plan_schema.py -q`
Expected: FAIL — `ImportError: cannot import name 'FieldSpec'`

- [ ] **Step 3: Extend the schema**

In `src/lottie/scaffold/schema.py`, change the `RenderInput.context` type and append the new models. The full file becomes:

```python
"""Typed contracts for template rendering and AI scaffolding.

The skill boundary is fully typed (CLAUDE.md rule 2); the dict handed to Jinja is
internal to the renderer.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RenderContext(BaseModel):
    """Variables injected into a scaffold template."""

    name: str
    class_name: str
    provider: str = "anthropic/claude-sonnet-4-6"


class RenderInput(BaseModel):
    """Input to TemplateRendererSkill — which template, with what context."""

    template: str
    context: BaseModel


class RenderOutput(BaseModel):
    """Rendered template content."""

    content: str


class FieldSpec(BaseModel):
    """One typed field on a generated Input/Output model."""

    name: str
    type: str
    description: str = ""


class ScaffoldRequest(BaseModel):
    """A natural-language request to generate an agent or skill."""

    kind: Literal["agent", "skill"]
    name: str
    description: str
    repair_feedback: str | None = None


class ScaffoldPlan(BaseModel):
    """Structured plan the LLM produces; renders into a unit module."""

    class_name: str
    input_fields: list[FieldSpec] = Field(min_length=1)
    output_fields: list[FieldSpec] = Field(min_length=1)
    system_prompt: str
    run_body: str
    tools: list[str] = []


class PlanRenderContext(BaseModel):
    """Full context for the `*_desc` Jinja templates."""

    name: str
    class_name: str
    provider: str
    kind: Literal["agent", "skill"]
    input_fields: list[FieldSpec]
    output_fields: list[FieldSpec]
    system_prompt: str
    run_body: str
    tools: list[str]
    input_sample: str


class ScaffoldResult(BaseModel):
    """Outcome of an AI scaffold run."""

    files_written: list[str]
    passed: bool
    diagnostics: str = ""
```

- [ ] **Step 4: Run tests to verify they pass (and existing renderer/generator still green)**

Run: `uv run pytest src/lottie/scaffold -q`
Expected: PASS (new plan-schema tests + existing renderer/generator tests)

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy --strict src/lottie/scaffold/schema.py && uv run ruff check src/lottie/scaffold/schema.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/lottie/scaffold/schema.py src/lottie/scaffold/tests/test_plan_schema.py
git commit -m "feat(scaffold): add ScaffoldPlan/Request schemas and plan render context"
```

---

## Task 10: `*_desc` Jinja templates for AI-generated units

**Files:**
- Create: `src/lottie/scaffold/templates/agent_desc/{AGENT.md.j2,agent.py.j2,schema.py.j2,config.yaml.j2,prompts.py.j2,test.py.j2}`
- Create: `src/lottie/scaffold/templates/skill_desc/{SKILL.md.j2,skill.py.j2,schema.py.j2,test.py.j2}`
- Test: `src/lottie/scaffold/tests/test_desc_templates.py`

- [ ] **Step 1: Write the failing test**

Create `src/lottie/scaffold/tests/test_desc_templates.py`:

```python
from __future__ import annotations

from lottie.scaffold.renderer import TemplateRendererSkill
from lottie.scaffold.schema import FieldSpec, PlanRenderContext, RenderInput


def _ctx(kind: str) -> PlanRenderContext:
    return PlanRenderContext(
        name="greeter",
        class_name="GreeterAgent" if kind == "agent" else "GreeterSkill",
        provider="anthropic/claude-sonnet-4-6",
        kind=kind,  # type: ignore[arg-type]
        input_fields=[FieldSpec(name="who", type="str")],
        output_fields=[FieldSpec(name="greeting", type="str")],
        system_prompt="Greet warmly.",
        run_body='return GreeterAgentOutput(greeting="hi")'
        if kind == "agent"
        else 'return GreeterSkillOutput(greeting="hi")',
        tools=["WebSearchSkill"],
        input_sample='who="x"',
    )


def test_agent_schema_template_renders_fields() -> None:
    skill = TemplateRendererSkill(enable_benchmarks=False)
    out = skill.run(RenderInput(template="agent_desc/schema.py.j2", context=_ctx("agent")))
    assert "who: str" in out.content
    assert "greeting: str" in out.content
    assert "class GreeterAgentInput(BaseModel)" in out.content


def test_agent_py_template_injects_run_body() -> None:
    skill = TemplateRendererSkill(enable_benchmarks=False)
    out = skill.run(RenderInput(template="agent_desc/agent.py.j2", context=_ctx("agent")))
    assert "class GreeterAgent(BaseAgent" in out.content
    assert "        return GreeterAgentOutput(greeting=\"hi\")" in out.content


def test_agent_config_lists_tools() -> None:
    skill = TemplateRendererSkill(enable_benchmarks=False)
    out = skill.run(RenderInput(template="agent_desc/config.yaml.j2", context=_ctx("agent")))
    assert "capabilities: [WebSearchSkill]" in out.content


def test_skill_py_template_injects_run_body() -> None:
    skill = TemplateRendererSkill(enable_benchmarks=False)
    out = skill.run(RenderInput(template="skill_desc/skill.py.j2", context=_ctx("skill")))
    assert "class GreeterSkill(BaseSkill" in out.content
    assert "        return GreeterSkillOutput(greeting=\"hi\")" in out.content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/scaffold/tests/test_desc_templates.py -q`
Expected: FAIL — `jinja2.exceptions.TemplateNotFound: agent_desc/schema.py.j2`

- [ ] **Step 3: Create the agent_desc templates**

`src/lottie/scaffold/templates/agent_desc/schema.py.j2`:

```jinja
"""Typed input/output models for {{ class_name }}."""
from __future__ import annotations
from pydantic import BaseModel


class {{ class_name }}Input(BaseModel):
    """Input for {{ class_name }}."""
{% for f in input_fields %}    {{ f.name }}: {{ f.type }}
{% endfor %}

class {{ class_name }}Output(BaseModel):
    """Output from {{ class_name }}."""
{% for f in output_fields %}    {{ f.name }}: {{ f.type }}
{% endfor %}
```

`src/lottie/scaffold/templates/agent_desc/agent.py.j2`:

```jinja
"""{{ class_name }} — generated by `lottie create agent {{ name }} --from-desc`."""
from __future__ import annotations
from lottie.core import BaseAgent
from lottie.llm import Message
from .prompts import SYSTEM_PROMPT
from .schema import {{ class_name }}Input, {{ class_name }}Output


class {{ class_name }}(BaseAgent[{{ class_name }}Input, {{ class_name }}Output]):
    """Generated agent. The body below was written by the ScaffolderAgent."""

    def _execute(self, data: {{ class_name }}Input) -> {{ class_name }}Output:
{{ run_body | indent(8, True) }}
```

`src/lottie/scaffold/templates/agent_desc/config.yaml.j2`:

```jinja
provider: {{ provider }}
model_params:
  temperature: 0.3
  max_tokens: 2048
capabilities: [{{ tools | join(", ") }}]
policies:
  - base
memory:
  enabled: false
  namespace: {{ name }}
```

`src/lottie/scaffold/templates/agent_desc/prompts.py.j2`:

```jinja
"""Prompt templates for {{ class_name }}."""
from __future__ import annotations

SYSTEM_PROMPT = """\
{{ system_prompt }}
"""
```

`src/lottie/scaffold/templates/agent_desc/AGENT.md.j2`:

```jinja
# {{ class_name }}

## Role
Generated from: {{ name }}

## Input
| Field | Type | Description |
|---|---|---|
{% for f in input_fields %}| {{ f.name }} | {{ f.type }} | {{ f.description }} |
{% endfor %}
## Output
| Field | Type | Description |
|---|---|---|
{% for f in output_fields %}| {{ f.name }} | {{ f.type }} | {{ f.description }} |
{% endfor %}
## Provider
Default: {{ provider }}

## Tools (Skills used)
{% for t in tools %}- {{ t }}
{% endfor %}
## Policies
- base
```

`src/lottie/scaffold/templates/agent_desc/test.py.j2`:

```jinja
"""Integration tests for {{ class_name }} (MockLLMProvider — no real LLM)."""
from __future__ import annotations
from lottie.llm import MockLLMProvider
from agents.{{ name }}.agent import {{ class_name }}
from agents.{{ name }}.schema import {{ class_name }}Input


def test_{{ name }}_runs() -> None:
    agent = {{ class_name }}(llm=MockLLMProvider(["mock response"]))
    result = agent.run({{ class_name }}Input({{ input_sample }}))
    assert result is not None
```

- [ ] **Step 4: Create the skill_desc templates**

`src/lottie/scaffold/templates/skill_desc/schema.py.j2`:

```jinja
"""Typed input/output models for {{ class_name }}."""
from __future__ import annotations
from pydantic import BaseModel


class {{ class_name }}Input(BaseModel):
    """Input for {{ class_name }}."""
{% for f in input_fields %}    {{ f.name }}: {{ f.type }}
{% endfor %}

class {{ class_name }}Output(BaseModel):
    """Output from {{ class_name }}."""
{% for f in output_fields %}    {{ f.name }}: {{ f.type }}
{% endfor %}
```

`src/lottie/scaffold/templates/skill_desc/skill.py.j2`:

```jinja
"""{{ class_name }} — generated by `lottie create skill {{ name }} --from-desc`."""
from __future__ import annotations
from lottie.core import BaseSkill
from .schema import {{ class_name }}Input, {{ class_name }}Output


class {{ class_name }}(BaseSkill[{{ class_name }}Input, {{ class_name }}Output]):
    """Generated skill. The body below was written by the ScaffolderAgent."""

    def _execute(self, data: {{ class_name }}Input) -> {{ class_name }}Output:
{{ run_body | indent(8, True) }}
```

`src/lottie/scaffold/templates/skill_desc/SKILL.md.j2`:

```jinja
# {{ class_name }}

## What it does
Generated from: {{ name }}

## Input
| Field | Type | Required | Description |
|---|---|---|---|
{% for f in input_fields %}| {{ f.name }} | {{ f.type }} | yes | {{ f.description }} |
{% endfor %}
## Output
| Field | Type | Description |
|---|---|---|
{% for f in output_fields %}| {{ f.name }} | {{ f.type }} | {{ f.description }} |
{% endfor %}
## Side effects
None.
```

`src/lottie/scaffold/templates/skill_desc/test.py.j2`:

```jinja
"""Unit tests for {{ class_name }} (deterministic — no LLM)."""
from __future__ import annotations
from skills.{{ name }}.schema import {{ class_name }}Input
from skills.{{ name }}.skill import {{ class_name }}


def test_{{ name }}_runs() -> None:
    skill = {{ class_name }}()
    result = skill.run({{ class_name }}Input({{ input_sample }}))
    assert result is not None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest src/lottie/scaffold/tests/test_desc_templates.py -q`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add src/lottie/scaffold/templates/agent_desc/ src/lottie/scaffold/templates/skill_desc/ src/lottie/scaffold/tests/test_desc_templates.py
git commit -m "feat(scaffold): add dynamic agent_desc/skill_desc templates"
```

---

## Task 11: ScaffolderAgent + prompt + plan parsing

**Files:**
- Create: `src/lottie/scaffold/scaffolder_prompts.py`
- Create: `src/lottie/scaffold/scaffolder_agent.py`
- Test: `src/lottie/scaffold/tests/test_scaffolder_agent.py`

- [ ] **Step 1: Write the failing test**

Create `src/lottie/scaffold/tests/test_scaffolder_agent.py`:

```python
from __future__ import annotations

import json

import pytest

from lottie.llm import MockLLMProvider
from lottie.scaffold.scaffolder_agent import ScaffolderAgent
from lottie.scaffold.schema import ScaffoldRequest

_PLAN = {
    "class_name": "GreeterAgent",
    "input_fields": [{"name": "who", "type": "str", "description": "name"}],
    "output_fields": [{"name": "greeting", "type": "str", "description": "msg"}],
    "system_prompt": "Greet warmly.",
    "run_body": 'return GreeterAgentOutput(greeting="hi")',
    "tools": [],
}


def test_returns_validated_plan() -> None:
    agent = ScaffolderAgent(llm=MockLLMProvider([json.dumps(_PLAN)]))
    plan = agent.run(ScaffoldRequest(kind="agent", name="greeter", description="greets"))
    assert plan.class_name == "GreeterAgent"
    assert plan.input_fields[0].name == "who"


def test_strips_code_fences() -> None:
    fenced = "```json\n" + json.dumps(_PLAN) + "\n```"
    agent = ScaffolderAgent(llm=MockLLMProvider([fenced]))
    plan = agent.run(ScaffoldRequest(kind="agent", name="greeter", description="greets"))
    assert plan.class_name == "GreeterAgent"


def test_repair_feedback_appears_in_prompt() -> None:
    mock = MockLLMProvider([json.dumps(_PLAN)])
    agent = ScaffolderAgent(llm=mock)
    agent.run(
        ScaffoldRequest(
            kind="agent", name="greeter", description="greets", repair_feedback="mypy: bad type"
        )
    )
    sent = mock.calls[0][-1].content
    assert "mypy: bad type" in sent


def test_invalid_json_raises() -> None:
    agent = ScaffolderAgent(llm=MockLLMProvider(["not json at all"]))
    with pytest.raises(ValueError, match="plan"):
        agent.run(ScaffoldRequest(kind="agent", name="greeter", description="greets"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/scaffold/tests/test_scaffolder_agent.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.scaffold.scaffolder_agent'`

- [ ] **Step 3: Write the prompt module**

Create `src/lottie/scaffold/scaffolder_prompts.py`:

```python
"""System and user prompts for the ScaffolderAgent."""

from __future__ import annotations

from lottie.scaffold.schema import ScaffoldRequest

SYSTEM_PROMPT = """\
You are the Lottie ScaffolderAgent. You generate a single JSON object describing
an agent or skill. Output ONLY the JSON — no prose, no markdown fences.

Rules you must never break:
1. All LLM calls use self.complete(...); never import openai or anthropic.
2. Inputs/outputs are the Pydantic models named <ClassName>Input / <ClassName>Output.
3. run_body is the BODY of _execute only — no def line, no decorators. Statements
   start at column 0; they will be indented for you. It must end by returning a
   <ClassName>Output(...) built from data.
4. For an agent, run_body MUST call self.complete([...]) building Message objects
   and reference SYSTEM_PROMPT, then return the Output from response.content.
5. For a skill, run_body must be deterministic — no LLM, no network.
6. Every field type is one of: str, int, float, bool, list[str].

JSON shape:
{
  "class_name": "PascalCase ending in Agent or Skill",
  "input_fields":  [{"name": "...", "type": "...", "description": "..."}],
  "output_fields": [{"name": "...", "type": "...", "description": "..."}],
  "system_prompt": "the agent's system prompt (ignored for skills)",
  "run_body": "python statements ...",
  "tools": ["SkillName", "..."]
}
"""


def build_user_prompt(request: ScaffoldRequest, class_name: str) -> str:
    """Compose the user message for one plan request, including any repair feedback."""
    lines = [
        f"Generate a Lottie {request.kind} named '{request.name}'.",
        f"Use class_name '{class_name}'.",
        f"Description: {request.description}",
    ]
    if request.repair_feedback:
        lines.append(
            "The previous attempt failed validation. Fix these problems:\n"
            + request.repair_feedback
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Write the agent**

Create `src/lottie/scaffold/scaffolder_agent.py`:

```python
"""ScaffolderAgent — turns a natural-language description into a ScaffoldPlan.

The agent only produces the validated plan via the injected LLMProvider; rendering,
the rule-13 gate, and writing live in `generator.generate_from_desc`. Tests inject
MockLLMProvider (CLAUDE.md rule 5).
"""

from __future__ import annotations

from lottie.core import BaseAgent
from lottie.llm import Message
from lottie.scaffold.generator import _class_name
from lottie.scaffold.scaffolder_prompts import SYSTEM_PROMPT, build_user_prompt
from lottie.scaffold.schema import ScaffoldPlan, ScaffoldRequest


class ScaffolderAgent(BaseAgent[ScaffoldRequest, ScaffoldPlan]):
    """Generate a structured ScaffoldPlan from a description."""

    def _execute(self, data: ScaffoldRequest) -> ScaffoldPlan:
        class_name = _class_name(data.name, data.kind)
        response = self.complete(
            [
                Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content=build_user_prompt(data, class_name)),
            ]
        )
        return _parse_plan(response.content)


def _parse_plan(content: str) -> ScaffoldPlan:
    """Parse the LLM response into a ScaffoldPlan, tolerating ```json fences."""
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.endswith("```"):
            text = text[: -len("```")]
        text = text.strip()
    try:
        return ScaffoldPlan.model_validate_json(text)
    except ValueError as exc:
        raise ValueError(f"ScaffolderAgent produced an invalid plan: {exc}") from exc
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest src/lottie/scaffold/tests/test_scaffolder_agent.py -q`
Expected: PASS (4 passed)

- [ ] **Step 6: Typecheck and lint**

Run: `uv run mypy --strict src/lottie/scaffold/scaffolder_agent.py src/lottie/scaffold/scaffolder_prompts.py && uv run ruff check src/lottie/scaffold/scaffolder_agent.py src/lottie/scaffold/scaffolder_prompts.py`
Expected: no errors

- [ ] **Step 7: Commit**

```bash
git add src/lottie/scaffold/scaffolder_agent.py src/lottie/scaffold/scaffolder_prompts.py src/lottie/scaffold/tests/test_scaffolder_agent.py
git commit -m "feat(scaffold): add ScaffolderAgent and plan parsing"
```

---

## Task 12: generate_from_desc orchestration (render + gate + retry)

**Files:**
- Modify: `src/lottie/scaffold/generator.py`
- Test: `src/lottie/scaffold/tests/test_generate_from_desc.py`

- [ ] **Step 1: Write the failing test**

Create `src/lottie/scaffold/tests/test_generate_from_desc.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lottie.llm import MockLLMProvider
from lottie.scaffold.generator import generate_from_desc

_GOOD_AGENT = {
    "class_name": "GreeterAgent",
    "input_fields": [{"name": "who", "type": "str", "description": "name"}],
    "output_fields": [{"name": "greeting", "type": "str", "description": "msg"}],
    "system_prompt": "Greet warmly.",
    "run_body": (
        "from lottie.llm import Message\n"
        "response = self.complete([Message(role='system', content=SYSTEM_PROMPT),"
        " Message(role='user', content=data.who)])\n"
        "return GreeterAgentOutput(greeting=response.content)"
    ),
    "tools": [],
}

_SECRET_SKILL = {
    "class_name": "LeakSkill",
    "input_fields": [{"name": "text", "type": "str", "description": "in"}],
    "output_fields": [{"name": "result", "type": "str", "description": "out"}],
    "system_prompt": "",
    "run_body": 'key = "AKIAIOSFODNN7EXAMPLE"\nreturn LeakSkillOutput(result=key)',
    "tools": [],
}


def _project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "lottie.yaml").write_text("project: demo\n", encoding="utf-8")
    (tmp_path / "agents").mkdir()
    (tmp_path / "skills").mkdir()
    return tmp_path


def test_clean_agent_is_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path, monkeypatch)
    llm = MockLLMProvider([json.dumps(_GOOD_AGENT)])
    result = generate_from_desc("agent", "greeter", "greets people", llm)
    assert result.passed is True
    assert (root / "agents" / "greeter" / "agent.py").is_file()


def test_secret_plan_is_rejected_and_rolled_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path, monkeypatch)
    # Same plan twice: initial attempt + the one retry both leak.
    llm = MockLLMProvider([json.dumps(_SECRET_SKILL), json.dumps(_SECRET_SKILL)])
    with pytest.raises(Exception):
        generate_from_desc("skill", "leak", "leaks a key", llm)
    assert not (root / "skills" / "leak").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/scaffold/tests/test_generate_from_desc.py -q`
Expected: FAIL — `ImportError: cannot import name 'generate_from_desc'`

- [ ] **Step 3: Add the orchestration to generator.py**

Add these imports to the top of `src/lottie/scaffold/generator.py` (alongside the existing ones):

```python
from lottie.llm import LLMProvider
from lottie.scaffold.scaffolder_agent import ScaffolderAgent
from lottie.scaffold.schema import (
    FieldSpec,
    PlanRenderContext,
    RenderContext,
    RenderInput,
    ScaffoldPlan,
    ScaffoldRequest,
    ScaffoldResult,
)
```

(Remove the now-duplicate `from lottie.scaffold.schema import RenderContext, RenderInput` line — fold it into the block above.)

Append to the end of `src/lottie/scaffold/generator.py`:

```python
_AGENT_DESC_PLAN: list[tuple[str, str]] = [
    ("__init__.py", ""),
    ("AGENT.md", "agent_desc/AGENT.md.j2"),
    ("agent.py", "agent_desc/agent.py.j2"),
    ("schema.py", "agent_desc/schema.py.j2"),
    ("config.yaml", "agent_desc/config.yaml.j2"),
    ("prompts.py", "agent_desc/prompts.py.j2"),
    ("tests/__init__.py", ""),
    ("tests/test_{name}.py", "agent_desc/test.py.j2"),
]
_SKILL_DESC_PLAN: list[tuple[str, str]] = [
    ("__init__.py", ""),
    ("SKILL.md", "skill_desc/SKILL.md.j2"),
    ("skill.py", "skill_desc/skill.py.j2"),
    ("schema.py", "skill_desc/schema.py.j2"),
    ("tests/__init__.py", ""),
    ("tests/test_{name}.py", "skill_desc/test.py.j2"),
]
_DESC_PLANS: dict[Kind, list[tuple[str, str]]] = {
    "agent": _AGENT_DESC_PLAN,
    "skill": _SKILL_DESC_PLAN,
}

_SAMPLE: dict[str, str] = {
    "str": '""',
    "int": "0",
    "float": "0.0",
    "bool": "False",
    "list[str]": "[]",
}


def _input_sample(fields: list[FieldSpec]) -> str:
    """A keyword-args string constructing the Input with placeholder values."""
    return ", ".join(f"{f.name}={_SAMPLE.get(f.type, 'None')}" for f in fields)


def _plan_context(kind: Kind, name: str, plan: ScaffoldPlan) -> PlanRenderContext:
    return PlanRenderContext(
        name=name,
        class_name=plan.class_name,
        provider=RenderContext(name=name, class_name=plan.class_name).provider,
        kind=kind,
        input_fields=plan.input_fields,
        output_fields=plan.output_fields,
        system_prompt=plan.system_prompt,
        run_body=plan.run_body,
        tools=plan.tools,
        input_sample=_input_sample(plan.input_fields),
    )


def _render_from_plan(kind: Kind, name: str, plan: ScaffoldPlan) -> dict[str, str]:
    context = _plan_context(kind, name, plan)
    renderer = TemplateRendererSkill(enable_benchmarks=False)
    files: dict[str, str] = {}
    for relpath, template in _DESC_PLANS[kind]:
        out = relpath.format(name=name)
        if not template:
            files[out] = ""
            continue
        try:
            files[out] = renderer.run(RenderInput(template=template, context=context)).content
        except TemplateError as exc:
            raise typer.BadParameter(f"failed to render {template}: {exc}") from exc
    return files


def generate_from_desc(
    kind: Kind, name: str, description: str, llm: LLMProvider
) -> ScaffoldResult:
    """AI-scaffold a unit from a description, behind the rule-13 write gate.

    One repair retry: gate diagnostics are fed back to the LLM before giving up.
    """
    from lottie.security import guard_and_write  # local import: avoid cycle at import time

    _validate_name(name)
    root = _project_root()
    parent_dir, _ = _PLANS[kind]
    target = root / parent_dir / name
    _guard(target)

    agent = ScaffolderAgent(llm=llm, enable_benchmarks=False)
    request = ScaffoldRequest(kind=kind, name=name, description=description)
    feedback: str | None = None
    last = None
    for _attempt in range(2):
        plan = agent.run(request.model_copy(update={"repair_feedback": feedback}))
        files = _render_from_plan(kind, name, plan)
        last = guard_and_write(target, files, root)
        if last.passed:
            _update_lottie_md(root, kind, plan.class_name, f"{parent_dir}/{name}/")
            return ScaffoldResult(
                files_written=last.files_written, passed=True, diagnostics=last.diagnostics
            )
        feedback = _format_feedback(last)
    raise typer.BadParameter(
        f"AI generation failed the security/validation gate after retry:\n{feedback}"
    )


def _format_feedback(result: object) -> str:
    """Render a GateResult into prompt-ready repair feedback."""
    from lottie.security.schema import GateResult

    assert isinstance(result, GateResult)
    lines: list[str] = []
    for f in result.findings:
        lines.append(f"security: {f.kind} at {f.file}:{f.line} — {f.message}")
    if result.diagnostics:
        lines.append(result.diagnostics)
    return "\n".join(lines) or "unknown gate failure"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/lottie/scaffold/tests/test_generate_from_desc.py -q`
Expected: PASS (2 passed). Note: the clean-agent case runs real mypy/ruff/bandit on the generated files — allow a few seconds.

- [ ] **Step 5: Typecheck, lint, full scaffold + security suite**

Run: `uv run mypy --strict src/lottie/scaffold && uv run ruff check src/lottie/scaffold && uv run pytest src/lottie/scaffold src/lottie/security -q`
Expected: no errors; all pass

- [ ] **Step 6: Commit**

```bash
git add src/lottie/scaffold/generator.py src/lottie/scaffold/tests/test_generate_from_desc.py
git commit -m "feat(scaffold): add generate_from_desc with rule-13 gate and retry"
```

---

## Task 13: Wire `--from-desc` into the CLI (deliverables #3, #4)

**Files:**
- Modify: `src/lottie/cli/create.py`
- Test: `src/lottie/cli/tests/test_create.py` (add cases)

- [ ] **Step 1: Write the failing test (append to existing file)**

Add to `src/lottie/cli/tests/test_create.py`:

```python
import json
from unittest.mock import patch

from lottie.llm import MockLLMProvider

_DESC_AGENT_PLAN = {
    "class_name": "GreeterAgent",
    "input_fields": [{"name": "who", "type": "str", "description": "name"}],
    "output_fields": [{"name": "greeting", "type": "str", "description": "msg"}],
    "system_prompt": "Greet warmly.",
    "run_body": (
        "from lottie.llm import Message\n"
        "response = self.complete([Message(role='system', content=SYSTEM_PROMPT),"
        " Message(role='user', content=data.who)])\n"
        "return GreeterAgentOutput(greeting=response.content)"
    ),
    "tools": [],
}


def test_create_agent_from_desc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    demo = _init_demo(tmp_path)
    monkeypatch.chdir(demo)
    fake_llm = MockLLMProvider([json.dumps(_DESC_AGENT_PLAN)])
    with patch("lottie.cli.create.build_provider", return_value=fake_llm):
        result = runner.invoke(
            app, ["create", "agent", "greeter", "--from-desc", "greets people"]
        )
    assert result.exit_code == 0, result.output
    agent_py = (demo / "agents" / "greeter" / "agent.py").read_text()
    assert "class GreeterAgent(BaseAgent" in agent_py
    assert "who: str" in (demo / "agents" / "greeter" / "schema.py").read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/cli/tests/test_create.py -q -k from_desc`
Expected: FAIL — `--from-desc` not a known option (exit code != 0 for the wrong reason / no such option)

- [ ] **Step 3: Add the option to the CLI**

Replace `src/lottie/cli/create.py` with:

```python
"""`lottie create agent|skill <name>` — scaffold a unit from templates or a description."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from lottie.llm import build_provider
from lottie.project.config import find_project_root, load_lottie_config
from lottie.scaffold.generator import generate, generate_from_desc

create_app = typer.Typer(help="Scaffold a new agent or skill.", no_args_is_help=True)

_FromDesc = Annotated[
    str | None,
    typer.Option("--from-desc", help="AI-generate the unit from this description."),
]


@create_app.command("agent")
def create_agent(name: str, from_desc: _FromDesc = None) -> None:
    """Scaffold an agent — from templates, or AI-generated with --from-desc."""
    _create("agent", name, from_desc)


@create_app.command("skill")
def create_skill(name: str, from_desc: _FromDesc = None) -> None:
    """Scaffold a skill — from templates, or AI-generated with --from-desc."""
    _create("skill", name, from_desc)


def _create(kind: str, name: str, from_desc: str | None) -> None:
    from typing import cast

    unit_kind = cast("Literal['agent', 'skill']", kind)
    if from_desc is None:
        target = generate(unit_kind, name)
        rel = target.relative_to(Path.cwd())
        typer.echo(f"Created {kind} at {rel}")
        typer.echo(f"Next: implement {rel / f'{kind}.py'} then run `pytest {rel}`")
        return
    root = find_project_root()
    llm = build_provider(load_lottie_config(root).providers.default)
    result = generate_from_desc(unit_kind, name, from_desc, llm)
    location = f"{kind}s/{name}/"
    typer.echo(f"AI-generated {kind} at {location} ({len(result.files_written)} files)")
```

Add the missing `Literal` import at the top:

```python
from typing import Annotated, Literal
```

(and drop the inline `from typing import cast` into the top block too: `from typing import Annotated, Literal, cast`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/lottie/cli/tests/test_create.py -q`
Expected: PASS (existing template tests + the new from-desc test)

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy --strict src/lottie/cli/create.py && uv run ruff check src/lottie/cli/create.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/lottie/cli/create.py src/lottie/cli/tests/test_create.py
git commit -m "feat(cli): add --from-desc AI generation to create agent/skill"
```

---

## Task 14: Close out — update checklist and full verification

**Files:**
- Modify: `LOTTIE_PHASE0_SPEC.md`

- [ ] **Step 1: Full verification suite (what CI runs)**

Run: `uv run ruff check . && uv run mypy --strict src && uv run pytest -q`
Expected: all green. If anything fails, fix before continuing — do not edit the checklist over a red suite.

- [ ] **Step 2: Mark the five deliverables done**

In `LOTTIE_PHASE0_SPEC.md` §8 Phase 0 Deliverables Checklist, update the five lines to:

```markdown
- [x] `lottie init` creates a valid project with hello-world agent
- [x] CI/CD: mypy, ruff, pytest on every push
- [x] `ScaffolderAgent` — the AI-powered generator agent
- [x] `TemplateRendererSkill`, `SchemaValidatorSkill` built-in skills
- [x] `LOTTIE.md` auto-updated by `lottie status`
```

Remove the earlier italic deferral notes on those lines. Leave the "pip install" line as-is (PyPI publish is out of scope for this plan).

- [ ] **Step 3: Commit**

```bash
git add LOTTIE_PHASE0_SPEC.md
git commit -m "docs: mark phase 0 deliverables complete"
```

- [ ] **Step 4: Confirm the demo works end-to-end (manual smoke)**

Run:
```bash
cd /tmp && rm -rf lottie_smoke && uv run --project /Users/cdiaz19/Documents/trae_projects/lottie-orchestrator lottie init lottie_smoke && ls lottie_smoke/agents/hello
```
Expected: hello agent files listed. (Running `lottie run hello` for real needs an API key; the integration test already covers it with a mocked provider.)

---

## Self-review notes (planner)

- **Spec coverage:** #1 hello-world → Task 7; #2 CI → Task 8; #3 ScaffolderAgent → Tasks 9–13; #4 SchemaValidatorSkill (+ security skills the gate needs) → Tasks 1–5; #5 status→LOTTIE.md → Task 6. Rule-13 full pipeline → Tasks 2–5. Hybrid generation (plan + run_body) → Tasks 9–12. 1-retry loop → Task 12. CI A1+B2 (main only, coverage report-only) → Task 8.
- **Deviations from spec (intentional, noted above):** `SchemaValidatorSkill` in `security/` not `scaffold/` (cycle avoidance); gate writes-then-rolls-back instead of temp-dir (mypy import resolution); built-in skills documented via docstring, not `SKILL.md`.
- **Type consistency:** `guard_and_write(target, files, root)`, `generate_from_desc(kind, name, description, llm)`, `_class_name(name, kind)`, `sync(root)`, `ScaffoldPlan`/`PlanRenderContext` field names are used identically across tasks.
