# test_retriever.py
# Integration tests for the Retriever orchestrator.
# Owner: Michael Ramirez

import pytest
from src.chunker import Chunk
from src.retriever import Retriever


class TestRetriever:
    def _make_chunk(self, text: str, token_count: int, index: int) -> Chunk:
        return Chunk(text=text, token_count=token_count, source="test.md", index=index)

    def test_query_before_load_raises(self):
        retriever = Retriever(token_budget=512)
        with pytest.raises(RuntimeError):
            retriever.query("some query")

    def test_load_chunks_and_query_returns_list(self):
        # TODO: implement once Retriever.query() is done
        raise NotImplementedError

    def test_results_within_token_budget(self):
        raise NotImplementedError

    def test_empty_corpus_returns_empty(self):
        raise NotImplementedError
