"""Fail-closed content gate for memory writes (CLAUDE.md rules 8/9/10).

Content headed for the memory store is screened with the same scanners that guard
serve I/O: oversize/control-char sanitize, prompt-injection scan, secret scan. Any
trip raises MemoryContentRejected — the write never happens. This is the memory-
poisoning defense: a run must not be able to write instructions or secrets that
hijack or exfiltrate from future runs. Messages never echo the offending content.

The screen itself lives in `content_gate.ContentGate` — distilled-skill drafts (V2 S3b)
need the identical three-scanner pass, and duplicating it would mean two execution paths
that could drift apart on a security boundary. This module keeps the memory-specific
name and exception type, which are what the rest of the codebase imports.

Imports only lottie.security — never memory/core, so security stays a leaf.
"""

from __future__ import annotations

from lottie.security.content_gate import ContentGate, ContentRejected


class MemoryContentRejected(ContentRejected):
    """Raised when content fails a memory-write security check. Carries no content."""


class MemoryContentGate(ContentGate):
    """Detect-and-block screen over content before it enters the memory store."""

    def __init__(self) -> None:
        super().__init__(
            source="memory-write",
            error=MemoryContentRejected,
            label="memory write",
        )
