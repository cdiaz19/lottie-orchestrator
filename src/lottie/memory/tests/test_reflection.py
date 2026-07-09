from lottie.memory.reflection import (
    RunTrajectory,
    build_reflection_prompt,
    parse_reflection,
)
from lottie.memory.schema import DeltaOp


def _traj() -> RunTrajectory:
    return RunTrajectory(
        task='{"q": "sum 2+2"}',
        outcome='{"a": "4"}',
        success=True,
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
        latency_ms=12.0,
    )


def test_prompt_has_system_and_user_with_trajectory() -> None:
    msgs = build_reflection_prompt(_traj())
    assert [m.role for m in msgs] == ["system", "user"]
    assert "sum 2+2" in msgs[1].content
    assert "success" in msgs[1].content.lower()


def test_parse_reflection_one_add_per_line() -> None:
    deltas = parse_reflection("check units before returning\n\nprefer int division here\n")
    assert len(deltas) == 2
    assert all(d.op is DeltaOp.ADD for d in deltas)
    assert deltas[0].content == "check units before returning"
    assert deltas[0].tags == ["reflection"]


def test_parse_reflection_empty_is_no_deltas() -> None:
    assert parse_reflection("   \n\n") == []
