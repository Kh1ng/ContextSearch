# test_search.py
# Unit tests for A* search.
# Owner: Michael Ramirez

import pytest
from src.chunker import Chunk
from src.search import astar_search, _keyword_overlap


class TestKeywordOverlap:
    def test_full_overlap(self):
        assert _keyword_overlap({"cat", "dog"}, {"cat", "dog"}) == 1.0

    def test_no_overlap(self):
        assert _keyword_overlap({"cat"}, {"dog"}) == 0.0

    def test_partial_overlap(self):
        score = _keyword_overlap({"cat", "fish"}, {"cat", "dog", "bird"})
        assert score == pytest.approx(1 / 3)

    def test_empty_query_keywords(self):
        assert _keyword_overlap({"cat"}, set()) == 0.0


class TestAstarSearch:
    def _make_chunk(self, text: str, token_count: int, keywords: set[str], index: int) -> Chunk:
        c = Chunk(text=text, token_count=token_count, source="test", index=index)
        c.keywords = keywords
        return c

    def test_returns_list(self):
        chunk = self._make_chunk("cats are mammals", 10, {"cats", "mammals"}, 0)
        result = astar_search("cats", [chunk], token_budget=100)
        assert isinstance(result, list)

    def test_respects_token_budget(self):
        chunks = [
            self._make_chunk("cats eat fish", 50, {"cats", "fish"}, 0),
            self._make_chunk("cats sleep a lot", 60, {"cats", "sleep"}, 1),
            self._make_chunk("cats like warm places", 40, {"cats", "warm"}, 2),
        ]
        result = astar_search("cats", chunks, token_budget=80)
        total_tokens = sum(c.token_count for c in result)
        assert total_tokens <= 80

    def test_prefers_high_relevance_chunks(self):
        low_relevance = self._make_chunk("dogs are loud", 10, {"dogs"}, 0)
        high_relevance = self._make_chunk("cats and kittens", 10, {"cats", "kittens"}, 1)
        result = astar_search("cats kittens", [low_relevance, high_relevance], token_budget=100)
        # High-relevance chunk should appear before low-relevance
        assert result[0] is high_relevance

    def test_empty_corpus_returns_empty(self):
        result = astar_search("cats", [], token_budget=100)
        assert result == []

    def test_single_chunk_within_budget(self):
        chunk = self._make_chunk("cats are cool", 20, {"cats"}, 0)
        result = astar_search("cats", [chunk], token_budget=100)
        assert result == [chunk]

    def test_single_chunk_exceeds_budget(self):
        chunk = self._make_chunk("cats are cool", 200, {"cats"}, 0)
        result = astar_search("cats", [chunk], token_budget=100)
        assert result == []
