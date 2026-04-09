# Design

## Problem

AI agent frameworks (Claude Code, OpenClaw, etc.) load instruction files and skills as context on every request. As these corpora grow, injecting everything is impractical — token costs scale linearly with corpus size and performance degrades when irrelevant context is included.

**Goal**: given a query and a corpus of documents, select the subset of chunks that maximizes relevance to the query while staying within a token budget.

---

## Algorithm: A\* Search

We frame chunk selection as a best-first search problem.

### State space

- Each **node** represents a candidate chunk.
- The **search frontier** is a priority queue of chunks, ordered by f(n).
- We iteratively select chunks to add to the context window until the token budget is exhausted.

### Cost function

```
f(n) = g(n) + h(n)
```

| Term | Meaning | Implementation |
|------|---------|----------------|
| `g(n)` | Cost to include this chunk | Token count of the chunk |
| `h(n)` | Estimated relevance | Keyword overlap score with query |
| `f(n)` | Priority score (lower = higher priority) | `g(n) - h(n)` — we invert h to prefer high-relevance, low-cost chunks |

> **Why A\* and not greedy?** Pure greedy search (maximize h alone) ignores token cost, so it may select a few large high-relevance chunks and exhaust the budget before including several smaller but collectively more relevant chunks. A\* balances both.

> **Why not BFS/Dijkstra?** These ignore relevance entirely. Our heuristic (keyword overlap) is admissible in the sense that it never overestimates relative value, making A\* both correct and efficient here.

### Heuristic: keyword overlap

```python
h(chunk, query) = |keywords(chunk) ∩ keywords(query)| / |keywords(query)|
```

Where `keywords()` strips stopwords and punctuation. This is a simple, interpretable baseline — future work could substitute TF-IDF weights, BM25, or dense embeddings.

---

## Chunking Strategy

Chunking is the most consequential design decision. A chunk that splits a key sentence in half is worse than no chunking at all.

### Strategies (in order of implementation priority)

1. **Fixed-size by token count** — split every N tokens with optional overlap. Simple, predictable, consistent budget accounting.
2. **Sentence boundary** — split on sentence endings. Preserves semantic units at the cost of variable chunk size.
3. **Markdown section** — split on `##` headings. Natural for agent instruction files; preserves topic coherence.

The default strategy is **fixed-size (256 tokens, 32-token overlap)**. The chunking strategy is configurable at `Retriever` initialization.

### Overlap

Overlapping chunks (e.g., last 32 tokens of chunk N are repeated as first 32 tokens of chunk N+1) prevent key information from being split across a boundary. This increases total token count slightly but improves retrieval recall.

---

## Token Counting

We use `tiktoken` with the `cl100k_base` encoding (GPT-4 / Claude-compatible). Token counts are computed at chunk-creation time and stored on the chunk object to avoid recomputation during search.

---

## Retriever Orchestration

`retriever.py` is the main entry point:

1. **Load corpus** — read files, chunk them, store chunks with metadata (source file, position, token count).
2. **Index** — precompute keyword sets for all chunks.
3. **Query** — run A\* over the chunk set, return ordered list of selected chunks up to budget.

---

## Open Questions / Future Work

- **Heuristic quality**: keyword overlap is a weak proxy for semantic relevance. BM25 or a small embedding model would substantially improve recall without adding much latency.
- **Chunking for code**: code files have different natural boundaries (functions, classes) than prose markdown.
- **Dynamic budget**: the token budget could be set as a fraction of the model's context window rather than a fixed number.
- **Deduplication**: overlapping chunks may introduce near-duplicate content into the context window; a dedup pass after selection could help.
