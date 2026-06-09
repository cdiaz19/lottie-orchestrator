# ResearchAgent

## Role
Knowledge-grounded Q&A agent that retrieves relevant chunks from the Lottie knowledge layer via `RetrievalSkill`, reasons over them with an LLM, and returns a concise digest with citations.

## Input

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | — | The research question to answer |
| `k` | `int` | `5` | Maximum number of chunks to retrieve |
| `layers` | `list[KnowledgeLayer]` | `[]` | Restrict retrieval to these layers (empty = all layers) |
| `expand_graph` | `bool` | `True` | Whether to expand hits via `depends_on` graph edges |

## Output

| Field | Type | Description |
|---|---|---|
| `digest` | `str` | Concise prose answer grounded in retrieved context |
| `points` | `list[str]` | Bullet-point highlights extracted by SummarizerSkill |
| `citations` | `list[Citation]` | Source references: `doc_id`, `chunk_id`, `score`, `source` |

## How Retrieval Works

1. `ResearchAgent._execute` builds a `RetrievalQuery(text, k, layers, expand_graph)`.
2. The query is passed to `RetrievalSkill.run(RetrievalSkillInput(query=rq))`.
3. `RetrievalSkill` embeds the query via its `EmbeddingProvider`, queries the `VectorStore`, and (when `expand_graph=True` and a `GraphStore` is present) appends discounted chunks from `depends_on` neighbours.
4. **The agent NEVER imports or touches the vector store or graph directly** — it only reads `result.hits` from the skill's output (Golden Rule, CLAUDE.md §11).

## How Summarisation Works

- The LLM response from `self.complete` is passed to `SummarizerSkill.run(SummarizerInput(text, max_points=k))`.
- `SummarizerSkill` makes its own LLM call to extract a prose `summary` and `points` bullets.
- When no `summarizer` is injected the agent constructs one from its own `llm`.

## Token Accumulation

All LLM calls in `_execute` go through `self.complete(messages)`, which forwards to `self.llm.complete` and then calls `self._active_ctx.add_usage(response.usage, response.cost_usd)`.  This means every call is automatically tracked in `agent.last_metrics` after `run()` completes.  The SummarizerSkill's internal LLM call is NOT accumulated (skills lack the RunContext accumulator — known Phase 2 gap).

## Grounded-Only Policy

The system prompt instructs the model to answer **only** from the numbered context passages.  If no knowledge is found, the context reads "No relevant knowledge found." and the model states that clearly rather than inventing an answer.

## Provider
Default: `anthropic/claude-sonnet-4-6`

## Skills Used (capabilities)
- `retrieval` — `RetrievalSkill(embedder, store, graph)` — must be injected at construction time
- `summarizer` — `SummarizerSkill(llm)` — injected or auto-constructed from agent LLM

## Policies
- `base`

## Examples

### Example 1 — basic query
```python
agent = ResearchAgent(llm, retrieval=retrieval_skill, summarizer=summarizer_skill)
out = agent.run(ResearchInput(query="What are multi-agent AI systems?", k=5))
print(out.digest)
print(out.citations)
```

### Example 2 — layer-filtered query
```python
out = agent.run(ResearchInput(
    query="platform auth conventions",
    k=3,
    layers=[KnowledgeLayer.PLATFORM],
    expand_graph=False,
))
```
