# search.py
# A* search over a set of candidate chunks.
# Owner: Michael Ramirez
#
# Given a query and a list of Chunks, returns the optimal subset of chunks
# that maximises heuristic relevance within a token budget.
#
# f(n) = g(n) + h(n)
#   g(n) = cumulative token cost of chunks selected so far
#   h(n) = heuristic score for this chunk (higher = more relevant)
#
# The heuristic is injected as a callable so the search loop is agnostic
# to whether relevance comes from keyword overlap or cosine similarity.
# Passing heuristic_fn=None falls back to keyword overlap.

import heapq
from collections.abc import Callable

from src.chunker import Chunk
from src.tokenizer import extract_keywords


def astar_search(
    query: str,
    chunks: list[Chunk],
    token_budget: int,
    heuristic_fn: Callable[[Chunk], float] | None = None,
) -> list[Chunk]:
    """
    Select chunks via A* to maximise relevance within token_budget.

    heuristic_fn(chunk) -> float in [0, 1]
        If None, defaults to keyword overlap against the query.

    Returns an ordered list of selected Chunk objects (highest relevance first).
    """
    if heuristic_fn is None:
        query_keywords = extract_keywords(query)
        heuristic_fn = lambda chunk: _keyword_overlap(chunk.keywords, query_keywords)

    heap = []
    for i, chunk in enumerate(chunks):
        relevance = heuristic_fn(chunk)

        # Skip chunks with no relevance signal — they add tokens without value.
        if relevance == 0.0:
            continue

        # f = token_count - relevance keeps low-cost, high-relevance chunks first.
        # i is a tiebreaker so the heap never compares Chunk objects directly.
        f = chunk.token_count - relevance
        heapq.heappush(heap, (f, i, chunk))

    selected = []
    remaining = token_budget

    while heap and remaining > 0:
        _f, _i, chunk = heapq.heappop(heap)
        if chunk.token_count <= remaining:
            selected.append(chunk)
            remaining -= chunk.token_count

    return selected


def _keyword_overlap(chunk_keywords: set[str], query_keywords: set[str]) -> float:
    """
    Heuristic h(n): fraction of query keywords present in chunk.
    Returns a value in [0, 1].
    """
    if not query_keywords:
        return 0.0
    return len(chunk_keywords & query_keywords) / len(query_keywords)
