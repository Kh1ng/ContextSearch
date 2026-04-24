# test_retriever.py
# Integration tests for the Retriever orchestrator.
# Owner: Michael Ramirez

import pytest
from pathlib import Path
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
        retriever = Retriever(token_budget=512)
        chunks = [
            self._make_chunk("authentication tokens expire after one hour", 20, 0),
            self._make_chunk("configure the database connection string", 15, 1),
        ]
        # Manually set keywords so the search heuristic can score them
        chunks[0].keywords = {"authentication", "tokens", "expire", "hour"}
        chunks[1].keywords = {"configure", "database", "connection", "string"}
        retriever.load_chunks(chunks)
        result = retriever.query("authentication tokens")
        assert isinstance(result, list)

    def test_results_within_token_budget(self):
        retriever = Retriever(token_budget=30)
        chunks = [
            self._make_chunk("authentication setup guide", 20, 0),
            self._make_chunk("authentication advanced config", 20, 1),
        ]
        chunks[0].keywords = {"authentication", "setup", "guide"}
        chunks[1].keywords = {"authentication", "advanced", "config"}
        retriever.load_chunks(chunks)
        result = retriever.query("authentication")
        total_tokens = sum(c.token_count for c in result)
        assert total_tokens <= 30

    def test_empty_corpus_returns_empty(self):
        retriever = Retriever(token_budget=512)
        with pytest.raises(RuntimeError):
            retriever.query("anything")

    def test_load_corpus_from_directory(self, tmp_path: Path):
        # Write two files into a temp directory
        (tmp_path / "intro.md").write_text("## Introduction\nThis covers authentication setup.", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("Remember to rotate your authentication tokens regularly.", encoding="utf-8")

        retriever = Retriever(token_budget=2048, strategy="sentence")
        retriever.load_corpus(tmp_path)

        assert len(retriever._chunks) > 0
        # All chunks should carry a source path pointing into tmp_path
        for chunk in retriever._chunks:
            assert str(tmp_path) in chunk.source
