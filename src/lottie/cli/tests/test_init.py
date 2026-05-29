from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.cli import app

runner = CliRunner()

EXPECTED_TREE = [
    "lottie.yaml",
    "LOTTIE.md",
    ".gitignore",
    "agents/__init__.py",
    "skills/__init__.py",
    "policies/base.yaml",
    "knowledge/global/.gitkeep",
    "knowledge/platform/.gitkeep",
    "knowledge/project/.gitkeep",
    "knowledge/memory/.gitkeep",
    "knowledge/draft/.gitkeep",
]


def test_app_exposes_init_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "init" in result.output
    # `init` must be reachable as a named subcommand, not just present in usage text.
    sub = runner.invoke(app, ["init", "--help"])
    assert sub.exit_code == 0


def test_init_creates_project_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "demo"])
    assert result.exit_code == 0, result.output

    root = tmp_path / "demo"
    for rel in EXPECTED_TREE:
        assert (root / rel).is_file(), f"missing {rel}"


def test_init_lottie_yaml_records_name_and_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "demo"])
    assert result.exit_code == 0, result.output
    text = (tmp_path / "demo" / "lottie.yaml").read_text()
    assert "project: demo" in text
    assert "default: anthropic/claude-sonnet-4-6" in text
    assert "fallback: openai/gpt-4o" in text


def test_init_gitignore_has_runtime_and_private_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "demo"])
    assert result.exit_code == 0, result.output
    text = (tmp_path / "demo" / ".gitignore").read_text()
    assert ".lottie/" in text
    assert ".private-journey/" in text


def test_init_base_policy_has_rule_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "demo"])
    assert result.exit_code == 0, result.output
    text = (tmp_path / "demo" / "policies" / "base.yaml").read_text()
    for key in ("allow:", "deny:", "escalate:"):
        assert key in text


def test_init_here_scaffolds_into_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "demo", "--here"])
    assert result.exit_code == 0, result.output

    # No nested ./demo/ subdir — files land directly in cwd.
    assert not (tmp_path / "demo").exists()
    assert (tmp_path / "lottie.yaml").is_file()
    assert "project: demo" in (tmp_path / "lottie.yaml").read_text()

    for rel in EXPECTED_TREE:
        assert (tmp_path / rel).is_file(), f"--here missing {rel}"
    # --here: user is already in the dir, so the Next-step hint must not contain `cd`.
    next_line = next(
        line for line in result.output.splitlines() if line.startswith("Next:")
    )
    assert "cd" not in next_line


def test_init_refuses_non_empty_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    existing = tmp_path / "demo"
    existing.mkdir()
    (existing / "keep.txt").write_text("important")

    result = runner.invoke(app, ["init", "demo"])

    assert result.exit_code != 0
    output = " ".join(result.output.split())
    assert "demo" in output
    assert "not empty" in output
    # Nothing clobbered, no scaffold written.
    assert (existing / "keep.txt").read_text() == "important"
    assert not (existing / "lottie.yaml").exists()


def test_init_here_refuses_existing_lottie_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "lottie.yaml").write_text("project: already-here\n")

    result = runner.invoke(app, ["init", "demo", "--here"])

    assert result.exit_code != 0
    output = " ".join(result.output.split())
    assert "lottie.yaml" in output
    # Existing config untouched.
    assert (tmp_path / "lottie.yaml").read_text() == "project: already-here\n"


def test_init_refuses_when_name_matches_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "demo").write_text("i am a file")

    result = runner.invoke(app, ["init", "demo"])

    # Must be a clean refusal, NOT a NotADirectoryError traceback.
    assert result.exit_code != 0
    output = " ".join(result.output.split())
    assert "demo" in output
    # The original file is untouched.
    assert (tmp_path / "demo").read_text() == "i am a file"


@pytest.mark.parametrize("bad_name", ["/tmp/evil", "../escape", "a/b", ".", "..", ""])
def test_init_rejects_invalid_project_names(
    bad_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", bad_name])
    assert result.exit_code != 0
    # No scaffold leaked outside or below cwd.
    assert not (tmp_path / "lottie.yaml").exists()
