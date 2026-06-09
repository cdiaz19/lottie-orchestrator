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
