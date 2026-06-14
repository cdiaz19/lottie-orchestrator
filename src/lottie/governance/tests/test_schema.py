from __future__ import annotations

from lottie.governance.schema import AuditRecord


def test_audit_record_minimal() -> None:
    r = AuditRecord(
        ts="2026-06-13T16:40:00+00:00",
        agent="echo",
        provider="mock/x",
        status="ok",
        root=True,
        input_sha256="a" * 64,
        output_sha256="b" * 64,
        input_tokens=1,
        output_tokens=2,
        cost_usd=0.001,
        latency_ms=12.5,
        error=None,
    )
    assert r.status == "ok" and r.root is True
    assert r.output_sha256 is not None


def test_audit_record_failure_has_no_output_hash() -> None:
    r = AuditRecord(
        ts="2026-06-13T16:40:00+00:00", agent="echo", provider=None, status="error",
        root=False, input_sha256="a" * 64, output_sha256=None, input_tokens=0,
        output_tokens=0, cost_usd=0.0, latency_ms=1.0, error="RuntimeError('boom')",
    )
    assert r.status == "error" and r.output_sha256 is None and r.error
