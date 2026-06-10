"""YAML frontmatter parser for knowledge documents.

Provides two public functions:

* ``parse_frontmatter`` — splits raw Markdown text into a metadata dict
  and a body string.  Never raises.
* ``to_document`` — wraps ``parse_frontmatter`` and constructs a
  :class:`~lottie.knowledge.schema.Document` from a file path, layer, and
  raw text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from lottie.knowledge.schema import DocStatus, Document, KnowledgeLayer


def parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Parse YAML frontmatter delimited by ``---`` fences.

    Parameters
    ----------
    text:
        Raw file content, potentially beginning with a ``---`` fence block.

    Returns
    -------
    tuple[dict[str, object], str]
        ``(metadata, body)`` where *metadata* is the parsed YAML dict (empty
        when absent, malformed, or not a mapping) and *body* is the content
        after the closing fence with any leading newline stripped.  The
        function **never raises**.
    """
    # Normalize CRLF to LF so Windows-authored files parse correctly.
    text = text.replace("\r\n", "\n")

    if not text.startswith("---"):
        return {}, text

    # Find the closing fence — must be on its own line after the opener.
    # Split off the opening "---\n", then look for the next "---" line.
    rest = text[3:]  # everything after the first "---"
    # The opener must be followed immediately by a newline (or EOF).
    if not rest.startswith("\n"):
        return {}, text

    rest_after_newline = rest[1:]  # skip the "\n" after opening "---"

    # Locate the closing fence.
    close_marker = "---"
    # Search line-by-line so we match the whole line, not an infix.
    lines = rest_after_newline.split("\n")
    close_index: int | None = None
    for i, line in enumerate(lines):
        if line.rstrip("\r") == close_marker:
            close_index = i
            break

    if close_index is None:
        # No closing fence found — no valid frontmatter.
        return {}, text

    yaml_text = "\n".join(lines[:close_index])
    body_lines = lines[close_index + 1 :]
    body = "\n".join(body_lines)
    # Strip exactly one leading newline that separates fence from content.
    body = body.removeprefix("\n")

    # Parse YAML safely — treat any exception or non-dict result as empty.
    try:
        parsed: Any = yaml.safe_load(yaml_text)
    except yaml.YAMLError:
        return {}, body

    if not isinstance(parsed, dict):
        return {}, body

    # Narrow the type: yaml.safe_load returns dict[Any, Any]; we want
    # dict[str, object].  Discard any non-string keys defensively.
    meta: dict[str, object] = {str(k): v for k, v in parsed.items()}
    return meta, body


def to_document(path: Path, layer: KnowledgeLayer, raw: str) -> Document:
    """Build a :class:`~lottie.knowledge.schema.Document` from a file path.

    Parameters
    ----------
    path:
        Filesystem path of the source document (used for *source* and as the
        fallback *id* when no ``id`` key is present in the frontmatter).
    layer:
        The :class:`~lottie.knowledge.schema.KnowledgeLayer` this document
        belongs to.
    raw:
        Raw file content (YAML frontmatter + body).

    Returns
    -------
    Document
        A fully populated :class:`~lottie.knowledge.schema.Document`; never
        raises.
    """
    meta, body = parse_frontmatter(raw)

    # --- id ---
    doc_id = str(meta["id"]) if meta.get("id") is not None else path.stem

    # --- tags ---
    raw_tags = meta.get("tags")
    tags: list[str] = [str(t) for t in raw_tags] if isinstance(raw_tags, list) else []

    # --- depends_on ---
    raw_deps = meta.get("depends_on")
    depends_on: list[str] = (
        [str(d) for d in raw_deps] if isinstance(raw_deps, list) else []
    )

    # --- status ---
    raw_status = meta.get("status")
    status: DocStatus
    if isinstance(raw_status, str):
        try:
            status = DocStatus(raw_status)
        except ValueError:
            status = DocStatus.DRAFT
    else:
        status = DocStatus.DRAFT

    # --- frontmatter (all top-level keys coerced to str) ---
    frontmatter: dict[str, str] = {str(k): str(v) for k, v in meta.items()}

    return Document(
        id=doc_id,
        source=str(path),
        layer=layer,
        content=body,
        frontmatter=frontmatter,
        tags=tags,
        depends_on=depends_on,
        status=status,
    )
