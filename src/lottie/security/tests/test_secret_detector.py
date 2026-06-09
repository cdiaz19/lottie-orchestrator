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
