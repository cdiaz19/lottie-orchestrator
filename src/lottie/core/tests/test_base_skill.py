from pathlib import Path

import pytest
from pydantic import BaseModel

from lottie.core import BaseSkill, RunMetrics


class _In(BaseModel):
    x: int


class _Out(BaseModel):
    y: int


class AddOneSkill(BaseSkill[_In, _Out]):
    def _execute(self, data: _In) -> _Out:
        return _Out(y=data.x + 1)


class BoomSkill(BaseSkill[_In, _Out]):
    def _execute(self, data: _In) -> _Out:
        raise ValueError("boom")


def test_skill_runs_and_returns_typed_output() -> None:
    out = AddOneSkill(enable_benchmarks=False).run(_In(x=1))
    assert out.y == 2


def test_skill_records_metrics() -> None:
    skill = AddOneSkill(enable_benchmarks=False)
    skill.run(_In(x=5))
    m = skill.last_metrics
    assert isinstance(m, RunMetrics)
    assert m.name == "AddOneSkill"
    assert m.kind == "skill"
    assert m.provider is None
    assert m.success is True
    assert m.latency_ms >= 0
    assert m.input_tokens == 0


def test_skill_records_metrics_on_failure_and_reraises() -> None:
    skill = BoomSkill(enable_benchmarks=False)
    with pytest.raises(ValueError, match="boom"):
        skill.run(_In(x=1))
    m = skill.last_metrics
    assert m is not None
    assert m.success is False
    assert "boom" in (m.error or "")


def test_skill_writes_jsonl_when_enabled(tmp_path: Path) -> None:
    skill = AddOneSkill(enable_benchmarks=True, benchmarks_root=tmp_path)
    skill.run(_In(x=1))
    assert (tmp_path / ".lottie" / "benchmarks" / "AddOneSkill.jsonl").exists()


def test_skill_custom_name() -> None:
    skill = AddOneSkill(name="adder", enable_benchmarks=False)
    skill.run(_In(x=1))
    assert skill.last_metrics is not None
    assert skill.last_metrics.name == "adder"
