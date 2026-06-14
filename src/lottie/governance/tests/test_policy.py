from __future__ import annotations

from pathlib import Path

import pytest

from lottie.governance.policy import (
    NullPolicyGate,
    PolicyConfigError,
    PolicyDenied,
    PolicyEscalation,
    PolicyGate,
    build_policy_gate,
    load_policy,
)


def _write(root: Path, name: str, body: str) -> None:
    (root / "policies").mkdir(parents=True, exist_ok=True)
    (root / "policies" / f"{name}.yaml").write_text(body, encoding="utf-8")


def test_load_policy_populated(tmp_path: Path) -> None:
    _write(tmp_path, "base", "name: base\ndeny: [shell]\nallow: [http]\nescalate: [fs]\n")
    p = load_policy(tmp_path, "base")
    assert p.deny == ["shell"] and p.allow == ["http"] and p.escalate == ["fs"]


def test_load_policy_empty_file_is_empty(tmp_path: Path) -> None:
    _write(tmp_path, "base", "")
    p = load_policy(tmp_path, "base")
    assert p.allow == [] and p.deny == [] and p.escalate == []


def test_load_policy_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(PolicyConfigError):
        load_policy(tmp_path, "nope")


def test_load_policy_malformed_yaml_raises_config_error(tmp_path: Path) -> None:
    # Fail closed AND typed: a YAML parse error must surface as PolicyConfigError,
    # not a raw yaml.YAMLError (regression for Round-7 finding FG-1).
    _write(tmp_path, "base", "allow: [unclosed\n")
    with pytest.raises(PolicyConfigError):
        load_policy(tmp_path, "base")


def test_gate_clean_passes() -> None:
    PolicyGate(["http"], allow=set(), deny=set(), escalate=set()).check()  # no raise


def test_gate_deny_raises() -> None:
    with pytest.raises(PolicyDenied):
        PolicyGate(["shell"], allow=set(), deny={"shell"}, escalate=set()).check()


def test_gate_escalate_raises() -> None:
    with pytest.raises(PolicyEscalation):
        PolicyGate(["fs"], allow=set(), deny=set(), escalate={"fs"}).check()


def test_gate_deny_beats_escalate() -> None:
    with pytest.raises(PolicyDenied):
        PolicyGate(["x"], allow=set(), deny={"x"}, escalate={"x"}).check()


def test_gate_allow_whitelist_blocks_unlisted() -> None:
    with pytest.raises(PolicyDenied):
        PolicyGate(["http", "shell"], allow={"http"}, deny=set(), escalate=set()).check()


def test_gate_allow_superset_passes() -> None:
    PolicyGate(["http"], allow={"http", "fs"}, deny=set(), escalate=set()).check()


def test_null_gate_never_raises() -> None:
    NullPolicyGate().check()


def test_build_gate_no_policies_is_null(tmp_path: Path) -> None:
    assert isinstance(build_policy_gate(tmp_path, policies=[], capabilities=["x"]), NullPolicyGate)


def test_build_gate_merges_files(tmp_path: Path) -> None:
    _write(tmp_path, "a", "deny: [shell]\n")
    _write(tmp_path, "b", "escalate: [fs]\n")
    gate = build_policy_gate(tmp_path, policies=["a", "b"], capabilities=["fs"])
    with pytest.raises(PolicyEscalation):
        gate.check()
    gate2 = build_policy_gate(tmp_path, policies=["a", "b"], capabilities=["shell"])
    with pytest.raises(PolicyDenied):
        gate2.check()
