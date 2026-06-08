"""guard_and_write — the rule-13 code-write gate.

Writes rendered files into the target directory, then runs the pipeline in order
(SecretDetection -> CodeSecurityScan -> mypy -> ruff). On any failure the target
directory is removed so no partial or unsafe code survives. Scanners always receive
absolute paths so they are independent of the subprocess cwd; mypy still derives a
generated unit's package (and relative imports) by walking up its __init__.py parents.
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
    root used to derive scanner-friendly relative paths for `files_written`. Scanners
    always receive absolute paths so they are independent of the subprocess cwd.
    """
    written: list[Path] = []
    for relpath, content in files.items():
        path = target / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written.append(path.resolve())

    # Scanners always get absolute paths — they launch subprocesses whose cwd is the
    # inherited process cwd (project root), not `root`, so relative paths would break.
    abs_all = [str(p) for p in written]
    abs_py = [str(p) for p in written if p.suffix == ".py" and p.stat().st_size > 0]

    # files_written uses root-relative paths when possible for human-readable output.
    rel_all = [_rel(p, root) for p in written]

    secrets = SecretDetectionSkill(enable_benchmarks=False).run(ScanInput(paths=abs_all))
    bandit = CodeSecurityScanSkill(enable_benchmarks=False).run(ScanInput(paths=abs_py))
    validation = SchemaValidatorSkill(enable_benchmarks=False).run(
        ValidateInput(paths=abs_py)
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
        files_written=rel_all,
    )


def _rel(path: Path, root: Path) -> str:
    """Path relative to root when possible, else absolute — as a string for skills."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
