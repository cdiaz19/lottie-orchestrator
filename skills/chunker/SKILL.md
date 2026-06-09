# ChunkerSkill

`ChunkerSkill` is a deterministic recursive-character chunker that splits a `Document` into an ordered list of `Chunk` objects using a sliding window with boundary snapping: for each non-final window the algorithm scans a priority list of separators (`"\n\n"`, `"\n"`, `". "`, `" "`) and snaps the chunk boundary to the last natural separator found inside the window, so chunks align with paragraph breaks, sentence ends, or word breaks rather than arbitrary character offsets.

## Input

| Field | Type | Required | Description |
|---|---|---|---|
| `document` | `Document` | yes | Source document (any `KnowledgeLayer`); `content` is the text to chunk |
| `config` | `ChunkConfig` | no | Window size (default 1000), overlap (default 200), separator priority list |

## Output

| Field | Type | Description |
|---|---|---|
| `chunks` | `list[Chunk]` | Ordered chunks with `id` (`<doc_id>#<idx>`), `start`/`end` char offsets, `text`, and `metadata` containing `layer` and `doc_id` for downstream vector-store filtering |

## Guarantees

- **Deterministic:** identical input always produces identical output.
- **No LLM:** pure Python sliding-window algorithm; unit-testable without mocks.
- **No side effects:** read-only; does not write to disk, network, or knowledge store.
- **Metadata stamped:** every chunk carries `metadata["layer"]` and `metadata["doc_id"]` for downstream retrieval filtering.

## Examples

```python
from lottie.knowledge.chunking import ChunkConfig
from lottie.knowledge.schema import Document, KnowledgeLayer
from skills.chunker.schema import ChunkerInput
from skills.chunker.skill import ChunkerSkill

doc = Document(id="readme", source="README.md", layer=KnowledgeLayer.PROJECT, content="...")
result = ChunkerSkill().run(ChunkerInput(document=doc))
# result.chunks → list[Chunk], each with .start, .end, .text, .metadata
```
