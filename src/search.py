# search.py
# A* search over a set of candidate chunks.
# Owner: Michael Ramirez
#
# Given a query and a list of Chunks, returns the optimal subset of chunks
# that maximises keyword-overlap relevance within a token budget.
#
# f(n) = g(n) + h(n)
#   g(n) = cumulative token cost of chunks selected so far
#   h(n) = keyword overlap score between chunk and query (higher = more relevant)
#
# The priority queue is ordered by -h(n)/g(n) so that high-relevance,
# low-cost chunks are explored first.

import heapq
from src.chunker import Chunk
from src.tokenizer import extract_keywords


def astar_search(
    query: str,
    chunks: list[Chunk],
    token_budget: int,
) -> list[Chunk]:
    """
    Select chunks via A* to maximise relevance within token_budget.

    Returns an ordered list of selected Chunk objects (highest relevance first).
    """
    raise NotImplementedError


def _keyword_overlap(chunk_keywords: set[str], query_keywords: set[str]) -> float:
    """
    Heuristic h(n): fraction of query keywords present in chunk.
    Returns a value in [0, 1].
    """
    if not query_keywords:
        return 0.0
    return len(chunk_keywords & query_keywords) / len(query_keywords)
