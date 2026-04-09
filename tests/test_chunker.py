# test_chunker.py
# Unit tests for Chunker and Chunk dataclass.
# Owner: Colton Spurgin

import pytest
from src.chunker import Chunk, Chunker


class TestChunk:
    def test_chunk_has_required_fields(self):
        chunk = Chunk(text="hello world", token_count=2, source="test.md", index=0)
        assert chunk.text == "hello world"
        assert chunk.token_count == 2
        assert chunk.source == "test.md"
        assert chunk.index == 0
        assert isinstance(chunk.keywords, set)


class TestChunkerFixed:
    def setup_method(self):
        self.chunker = Chunker(strategy="fixed", chunk_size=10, overlap=2)

    def test_returns_list_of_chunks(self):
        # TODO: implement once Chunker.chunk() is done
        raise NotImplementedError

    def test_no_chunk_exceeds_size(self):
        raise NotImplementedError

    def test_overlap_content_carried_forward(self):
        raise NotImplementedError

    def test_empty_string_returns_empty_list(self):
        raise NotImplementedError


class TestChunkerMarkdown:
    def setup_method(self):
        self.chunker = Chunker(strategy="markdown")

    def test_splits_on_headings(self):
        raise NotImplementedError

    def test_single_section_no_split(self):
        raise NotImplementedError


class TestChunkerSentence:
    def setup_method(self):
        self.chunker = Chunker(strategy="sentence", chunk_size=50)

    def test_splits_on_sentence_boundary(self):
        raise NotImplementedError
