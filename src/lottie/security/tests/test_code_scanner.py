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
