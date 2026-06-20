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


def test_overflow_emits_with_overlap_and_reconstructs() -> None:
    gate = StreamingSecretGate()
    big = "x" * 9000  # > MAX_LINE (8192), no newline, all identifier chars
    out = list(gate.scan_stream([big]))
    assert len(out) >= 2                  # head emitted on overflow, tail at flush
    assert "".join(out) == big            # byte-for-byte reconstruction


def test_secret_in_overflow_buffer_raises_and_does_not_leak() -> None:
    gate = StreamingSecretGate()
    big = "x" * 9000 + _AKIA              # secret after a long no-newline run
    out: list[str] = []
    with pytest.raises(OutputSecurityViolation):
        for piece in gate.scan_stream([big]):
            out.append(piece)
    assert _AKIA not in "".join(out)      # the secret never streamed


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
