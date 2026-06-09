"""Framework utilities for the document ingest pipeline.

Provides:
- ``load_source`` — load raw text from a typed ``IngestSource``.
- ``make_draft_id`` — deterministic, filesystem-safe draft identifier derived
  from the source kind/value.  Used as the document ``id`` and as the stem of
  the draft ``.md`` file written under ``knowledge/draft/``.

No LLM, no network (URL ingest is explicitly deferred).  Both functions are
pure / deterministic: identical inputs always produce identical outputs.

Dependency note
---------------
``lottie.knowledge.ingest`` imports ``IngestSource`` from
``skills.document_ingest.schema``.  That module is a thin Pydantic data
container with no reverse imports, so the dependency is acyclic.  The
alternative (a Protocol) requires ``Literal`` kind matching that mypy cannot
satisfy for a ``Literal`` attribute narrowing on a Protocol instance — the
direct import is both simpler and more type-safe.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from skills.document_ingest.schema import IngestSource

# ---------------------------------------------------------------------------
# Source loader
# ---------------------------------------------------------------------------


def load_source(source: IngestSource) -> str:
    """Return the raw text for *source*.

    Dispatch rules
    --------------
    - ``kind == "text"`` → return ``source.value`` as-is.
    - ``kind == "file"`` → read ``Path(source.value)`` with UTF-8 encoding.
    - ``kind == "url"``  → raise ``NotImplementedError`` (deferred to a later phase).

    Raises
    ------
    NotImplementedError
        When ``source.kind == "url"``.
    FileNotFoundError
        When ``source.kind == "file"`` and the path does not exist.
    """
    if source.kind == "text":
        return source.value
    if source.kind == "file":
        return Path(source.value).read_text(encoding="utf-8")
    # url — deferred
    raise NotImplementedError(
        f"URL ingest is deferred to a later phase (got: {source.value!r})"
    )


# ---------------------------------------------------------------------------
# Draft-id / filename helpers
# ---------------------------------------------------------------------------

_UNSAFE = re.compile(r"[^\w\-]")  # keep word chars and hyphens


def _slugify(text: str, max_len: int = 48) -> str:
    """Convert *text* to a lowercase filesystem-safe slug."""
    slug = _UNSAFE.sub("_", text.lower()).strip("_")
    # Collapse runs of underscores
    slug = re.sub(r"_+", "_", slug)
    return slug[:max_len]


def make_draft_id(source: IngestSource) -> str:
    """Return a deterministic, filesystem-safe draft id for *source*.

    Rules
    -----
    - ``kind == "file"`` → ``"draft/<slugified-stem>"`` using ``Path.stem``.
    - ``kind == "text"`` → ``"draft/text_<sha1[:12]>"`` where the SHA-1 is
      computed over the UTF-8-encoded content.
    - ``kind == "url"``  → ``"draft/url_<sha1[:12]>"`` (consistent but URLs
      are rejected by ``load_source`` before this is needed in practice).

    The returned id is always prefixed with ``"draft/"`` to make the target
    layer explicit and to satisfy CLAUDE.md rule 12 (agents write only to
    ``knowledge/draft/``).
    """
    if source.kind == "file":
        stem = _slugify(Path(source.value).stem) or "file"
        return f"draft/{stem}"
    # text or url: hash the value
    sha1 = hashlib.sha1(source.value.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    prefix = "text" if source.kind == "text" else "url"
    return f"draft/{prefix}_{sha1}"


def draft_filename(draft_id: str) -> str:
    """Return the ``.md`` filename for a *draft_id* (strips the ``draft/`` prefix)."""
    stem = draft_id.removeprefix("draft/")
    return f"{stem}.md"
