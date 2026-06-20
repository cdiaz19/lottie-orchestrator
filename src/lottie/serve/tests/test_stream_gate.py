from __future__ import annotations

from collections.abc import Generator
from typing import cast

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


def test_early_close_does_not_flush_held_buffer() -> None:
    """Flush is after the for-loop, NOT in a finally — so a consumer that closes the generator
    early never gets the unverified buffered tail (no leak), and close() does not raise."""
    gate = StreamingSecretGate()
    gen = cast(
        Generator[str, None, None],
        gate.scan_stream(["clean line\n", "UNVERIFIED_TAIL_NO_NEWLINE"]),
    )
    first = next(gen)
    assert first == "clean line\n"
    gen.close()  # must NOT raise (a finally-flush would RuntimeError 'ignored GeneratorExit')
    with pytest.raises(StopIteration):
        next(gen)


def test_detect_secrets_keyword_secret_raises() -> None:
    """A secret flagged by detect-secrets (KeywordDetector), NOT a custom regex — proves the gate
    reuses the full scan_text, not just the bounded patterns."""
    gate = StreamingSecretGate()
    secret_line = "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n"
    with pytest.raises(OutputSecurityViolation):
        list(gate.scan_stream([secret_line]))


_LINE_SCOPED_PLUGINS = {
    "ArtifactoryDetector", "AWSKeyDetector", "AzureStorageKeyDetector", "BasicAuthDetector",
    "CloudantDetector", "DiscordBotTokenDetector", "GitHubTokenDetector", "GitLabTokenDetector",
    "Base64HighEntropyString", "HexHighEntropyString", "IbmCloudIamDetector", "IbmCosHmacDetector",
    "IPPublicDetector", "JwtTokenDetector", "KeywordDetector", "MailchimpDetector", "NpmDetector",
    "OpenAIDetector", "PrivateKeyDetector", "PypiTokenDetector", "SendGridDetector",
    "SlackDetector",
    "SoftlayerDetector", "SquareOAuthDetector", "StripeDetector", "TelegramBotTokenDetector",
    "TwilioKeyDetector",
}


def test_configured_plugins_are_line_scoped() -> None:
    """Soundness guard: the gate is sound ONLY if every detect-secrets plugin is line-scoped.
    Pin the configured set; a detect-secrets upgrade that changes it fails here -> forces a human
    re-review (and verification that any new plugin is still per-line)."""
    from detect_secrets.settings import default_settings

    with default_settings() as settings:
        assert set(settings.plugins) == _LINE_SCOPED_PLUGINS


# ---------------------------------------------------------------------------
# Regression tests — the two confirmed leak paths, now caught (not emitted)
# ---------------------------------------------------------------------------

def test_private_key_header_with_space_in_long_line_caught_not_split() -> None:
    """Regression: a secret containing a SPACE (PRIVATE KEY header) in a long line is caught whole,
    never emitted split. The old overflow path leaked it across a mid-line cut."""
    gate = StreamingSecretGate()
    out: list[str] = []
    line = "x" * 4000 + "-----BEGIN RSA PRIVATE KEY-----\n"
    with pytest.raises(OutputSecurityViolation):
        for piece in gate.scan_stream([line]):
            out.append(piece)
    assert "PRIVATE KEY" not in "".join(out)
    assert "BEGIN" not in "".join(out)


def test_keyword_secret_in_long_line_caught_not_split() -> None:
    """Regression: same-line keyword secret in a long line is caught whole, value never emitted."""
    gate = StreamingSecretGate()
    out: list[str] = []
    line = "x" * 4000 + "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n"
    with pytest.raises(OutputSecurityViolation):
        for piece in gate.scan_stream([line]):
            out.append(piece)
    assert "wJalr" not in "".join(out)


def test_unterminated_over_cap_line_fails_closed_no_partial_emit() -> None:
    """An unterminated line beyond _MAX_LINE is withheld fail-closed — nothing emitted partially
    (a partial emit could split a secret)."""
    gate = StreamingSecretGate()
    out: list[str] = []
    with pytest.raises(OutputSecurityViolation):
        for piece in gate.scan_stream(["x" * 70000]):  # > 64 KB, no newline
            out.append(piece)
    assert out == []  # NOTHING emitted before the fail-closed raise


def test_long_single_line_under_cap_buffers_and_emits_whole_at_flush() -> None:
    """A long but under-cap single line buffers and emits whole at flush (sound; format-level for a
    single-line response) — byte-for-byte."""
    gate = StreamingSecretGate()
    text = "y" * 5000  # under cap, no newline, clean
    assert "".join(gate.scan_stream([text])) == text
