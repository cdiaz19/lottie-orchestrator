"""Framework utilities for the document ingest pipeline.

Provides:
- ``IngestSource`` — Pydantic model describing a single content source.
- ``load_source`` — load raw text from a typed ``IngestSource``.
- ``make_draft_id`` — deterministic, filesystem-safe draft identifier derived
  from the source kind/value and content hash.  Used as the document ``id``
  and as the stem of the draft ``.md`` file written under ``knowledge/draft/``.

No LLM, no network (URL ingest is explicitly deferred).  ``load_source`` and
``make_draft_id`` are pure / deterministic: identical inputs always produce
identical outputs.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from lottie.knowledge.schema import KnowledgeLayer

# ---------------------------------------------------------------------------
# IngestSource model (lives in the framework — skills import from here)
# ---------------------------------------------------------------------------


class IngestSource(BaseModel):
    """A single source of content to ingest into the knowledge layer.

    Parameters
    ----------
    kind:
        ``"text"`` — raw string content; ``"file"`` — path on disk;
        ``"url"`` — remote URL (deferred, raises ``NotImplementedError``).
    value:
        The source payload: the raw text, a file path, or a URL.
    layer:
        Requested *eventual* layer after human promotion. Phase 1 always
        writes to ``KnowledgeLayer.DRAFT`` regardless of this value.
    """

    kind: Literal["file", "text", "url"]
    value: str
    layer: KnowledgeLayer = KnowledgeLayer.DRAFT


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


def make_draft_id(source: IngestSource, content: str) -> str:
    """Return a deterministic, filesystem-safe draft id for *source*.

    Rules
    -----
    - ``kind == "file"`` → ``"draft/<slugified-stem>-<sha1(content)[:8]>"``
      using ``Path.stem`` for the slug and the first 8 hex chars of the SHA-1
      of the file *content*.  This prevents collisions when two files share the
      same stem but carry different content.
    - ``kind == "text"`` → ``"draft/text_<sha1(content)[:12]>"`` where the
      SHA-1 is computed over the UTF-8-encoded content (same as before).
    - ``kind == "url"``  → ``"draft/url_<sha1(content)[:12]>"`` (consistent
      but URLs are rejected by ``load_source`` before this is needed in
      practice).

    The function is **idempotent**: re-ingesting identical content from the
    same source always produces the same id.

    The returned id is always prefixed with ``"draft/"`` to make the target
    layer explicit and to satisfy CLAUDE.md rule 12 (agents write only to
    ``knowledge/draft/``).
    """
    sha1 = hashlib.sha1(content.encode("utf-8"), usedforsecurity=False).hexdigest()
    if source.kind == "file":
        stem = _slugify(Path(source.value).stem) or "file"
        return f"draft/{stem}-{sha1[:8]}"
    # text or url: hash the content
    prefix = "text" if source.kind == "text" else "url"
    return f"draft/{prefix}_{sha1[:12]}"


def draft_filename(draft_id: str) -> str:
    """Return the ``.md`` filename for a *draft_id* (strips the ``draft/`` prefix)."""
    stem = draft_id.removeprefix("draft/")
    return f"{stem}.md"


def _today_iso() -> str:
    """Return today's UTC date as an ISO 8601 string."""
    return datetime.now(UTC).date().isoformat()
