from __future__ import annotations

from lottie.security.input_sanitizer import InputSanitizerSkill
from lottie.security.schema import SanitizeInput


def _run(content: str, **kw: object) -> tuple[bool, str]:
    out = InputSanitizerSkill().run(SanitizeInput(content=content, **kw))
    return out.ok, out.reason


def test_clean_text_passes() -> None:
    ok, reason = _run("Summarize this: hello world\twith tabs\nand newlines.")
    assert ok is True
    assert reason == ""


def test_oversized_is_rejected() -> None:
    ok, reason = _run("x" * 11, max_len=10)
    assert ok is False
    assert reason == "oversized"


def test_control_characters_rejected() -> None:
    ok, reason = _run("payload\x00with-nul")
    assert ok is False
    assert reason == "control-characters"


def test_tab_newline_carriage_return_allowed() -> None:
    ok, _ = _run("a\tb\nc\rd")
    assert ok is True
