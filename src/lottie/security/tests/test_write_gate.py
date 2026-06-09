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
