import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from lottie.core.metrics import (
    RunContext,
    RunMetrics,
    append_metrics,
    git_version,
)
from lottie.llm import TokenUsage


def _sample_metrics(name: str = "demo") -> RunMetrics:
    return RunMetrics(
        name=name,
        kind="skill",
        provider=None,
        timestamp=datetime.now(UTC),
        latency_ms=1.5,
        success=True,
        version=None,
    )


def test_run_metrics_defaults() -> None:
    m = _sample_metrics()
    assert m.input_tokens == 0
    assert m.output_tokens == 0
    assert m.cost_usd == 0.0
    assert m.retry_count == 0
    assert m.error is None


def test_run_context_accumulates_usage() -> None:
    ctx = RunContext()
    ctx.add_usage(TokenUsage(input_tokens=10, output_tokens=5), cost_usd=0.01)
    ctx.add_usage(TokenUsage(input_tokens=2, output_tokens=3), cost_usd=0.02)
    assert ctx.input_tokens == 12
    assert ctx.output_tokens == 8
    assert round(ctx.cost_usd, 4) == 0.03


def test_git_version_returns_hash_in_repo(tmp_path: Path) -> None:
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    def run(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(tmp_path), *args],
            check=True,
            env={**os.environ, **env},
            capture_output=True,
        )

    run("init", "-q")
    (tmp_path / "f.txt").write_text("hi")
    run("add", ".")
    run("commit", "-q", "-m", "init")
    expected = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert git_version(cwd=tmp_path) == expected


def test_git_version_returns_none_outside_repo(tmp_path: Path) -> None:
    assert git_version(cwd=tmp_path) is None


def test_append_metrics_writes_jsonl(tmp_path: Path) -> None:
    path = append_metrics(_sample_metrics("alpha"), root=tmp_path, enabled=True)
    assert path == tmp_path / ".lottie" / "benchmarks" / "alpha.jsonl"
    assert path is not None and path.exists()
    line = path.read_text().strip()
    assert json.loads(line)["name"] == "alpha"


def test_append_metrics_appends_second_line(tmp_path: Path) -> None:
    append_metrics(_sample_metrics("beta"), root=tmp_path, enabled=True)
    path = append_metrics(_sample_metrics("beta"), root=tmp_path, enabled=True)
    assert path is not None
    assert len(path.read_text().strip().splitlines()) == 2


def test_append_metrics_disabled_returns_none(tmp_path: Path) -> None:
    path = append_metrics(_sample_metrics("gamma"), root=tmp_path, enabled=False)
    assert path is None
    assert not (tmp_path / ".lottie").exists()
