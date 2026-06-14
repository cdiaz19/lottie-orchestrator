from __future__ import annotations

from lottie.security.output_validator import OutputValidationSkill
from lottie.security.schema import OutputCheckInput


def _run(content: str, **kw: object) -> tuple[bool, str]:
    out = OutputValidationSkill().run(OutputCheckInput(content=content, **kw))
    return out.ok, out.reason


def test_clean_output_passes() -> None:
    ok, reason = _run("A perfectly normal answer.")
    assert ok is True
    assert reason == ""


def test_empty_output_rejected() -> None:
    ok, reason = _run("   \n\t ")
    assert ok is False
    assert reason == "empty"


def test_oversized_output_rejected() -> None:
    ok, reason = _run("y" * 11, max_len=10)
    assert ok is False
    assert reason == "oversized"
