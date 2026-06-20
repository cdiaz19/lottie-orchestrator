from __future__ import annotations

import pytest

from lottie.serve.errors import OutputSecurityViolation
from lottie.serve.stream_gate import StreamingSecretGate

_AKIA = "AKIA" + "1234567890ABCDEF"


def test_multiline_all_clean_emits_byte_for_byte() -> None:
    gate = StreamingSecretGate()
    text = "first line\nsecond line\r\nthird no-newline-tail"
    # fed as one delta
    assert "".join(gate.scan_stream([text])) == text
    # fed split across many deltas -> same bytes
    deltas = ["fir", "st line\nsec", "ond line\r", "\nthird ", "no-newline-tail"]
    assert "".join(gate.scan_stream(deltas)) == text


def test_secret_on_a_line_raises_after_prior_clean_lines() -> None:
    gate = StreamingSecretGate()
    out: list[str] = []
    with pytest.raises(OutputSecurityViolation):
        for piece in gate.scan_stream([f"safe line one\nhere is {_AKIA} oops\nnever reached\n"]):
            out.append(piece)
    assert out == ["safe line one\n"]      # the clean line emitted
    assert _AKIA not in "".join(out)        # the secret line NEVER yielded


def test_secret_split_across_deltas_within_one_line_caught() -> None:
    gate = StreamingSecretGate()
    out: list[str] = []
    with pytest.raises(OutputSecurityViolation):
        for piece in gate.scan_stream(["AKIA0000", "00000000", "000A\n"]):  # one line, reassembled
            out.append(piece)
    assert "".join(out) == ""               # nothing emitted (the only line is the secret)


def test_flush_emits_final_partial_clean_line() -> None:
    gate = StreamingSecretGate()
    # no trailing newline -> the last line is emitted at flush
    result = "".join(gate.scan_stream(["a clean tail with no newline"]))
    assert result == "a clean tail with no newline"
