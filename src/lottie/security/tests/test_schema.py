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
