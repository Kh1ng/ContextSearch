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
        # TODO: implement once astar_search() is done
        raise NotImplementedError

    def test_respects_token_budget(self):
        raise NotImplementedError

    def test_prefers_high_relevance_chunks(self):
        raise NotImplementedError

    def test_empty_corpus_returns_empty(self):
        raise NotImplementedError

    def test_single_chunk_within_budget(self):
        raise NotImplementedError

    def test_single_chunk_exceeds_budget(self):
        raise NotImplementedError
