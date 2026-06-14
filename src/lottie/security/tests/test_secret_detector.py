from __future__ import annotations

import os
import tempfile
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


_AWS = "AKIA" + "1234567890ABCDEF"  # split so this file isn't itself flagged


def test_scan_text_flags_aws_key_with_source_label() -> None:
    findings = SecretDetectionSkill().scan_text(f"token = {_AWS}\n", source="serve-output")
    assert findings
    assert all(f.file == "serve-output" for f in findings)  # never leaks a temp path


def test_scan_text_clean_returns_no_findings() -> None:
    assert SecretDetectionSkill().scan_text("just a normal answer") == []


def test_scan_text_removes_temp_file() -> None:
    before = set(os.listdir(tempfile.gettempdir()))
    SecretDetectionSkill().scan_text(f"k={_AWS}")
    after = set(os.listdir(tempfile.gettempdir()))
    assert after <= before  # no leftover temp artifact


def test_scan_text_parity_with_file_path(tmp_path: Path) -> None:
    content = f"aws={_AWS}\n-----BEGIN PRIVATE KEY-----\n"
    f = tmp_path / "blob.txt"
    f.write_text(content, encoding="utf-8")
    skill = SecretDetectionSkill()
    via_file = skill._execute(ScanInput(paths=[str(f)])).findings
    via_text = skill.scan_text(content, source="serve-output")
    assert {(x.kind, x.line) for x in via_file} == {(x.kind, x.line) for x in via_text}
