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
        text = " ".join(f"w{i}" for i in range(30))
        chunks = self.chunker.chunk(text, source="doc.txt")
        assert isinstance(chunks, list)
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_no_chunk_exceeds_size(self):
        text = " ".join(f"w{i}" for i in range(55))
        chunks = self.chunker.chunk(text, source="doc.txt")
        assert chunks
        assert all(len(c.text.split()) <= self.chunker.chunk_size for c in chunks)

    def test_overlap_content_carried_forward(self):
        text = " ".join(f"w{i}" for i in range(25))
        chunks = self.chunker.chunk(text, source="doc.txt")
        assert len(chunks) >= 2

        first_words = chunks[0].text.split()
        second_words = chunks[1].text.split()
        assert first_words[-self.chunker.overlap :] == second_words[: self.chunker.overlap]

    def test_empty_string_returns_empty_list(self):
        assert self.chunker.chunk("", source="doc.txt") == []
        assert self.chunker.chunk("   ", source="doc.txt") == []


class TestChunkerMarkdown:
    def setup_method(self):
        self.chunker = Chunker(strategy="markdown")

    def test_splits_on_headings(self):
        text = """# Title
Intro

## Setup
Install dependencies.

## Usage
Run the retriever.
"""
        chunks = self.chunker.chunk(text, source="guide.md")
        assert len(chunks) == 3
        assert chunks[0].text.startswith("# Title")
        assert chunks[1].text.startswith("## Setup")
        assert chunks[2].text.startswith("## Usage")

    def test_single_section_no_split(self):
        text = "This is a simple markdown-ish document with no headings."
        chunks = self.chunker.chunk(text, source="single.md")
        assert len(chunks) == 1
        assert chunks[0].text == text


class TestChunkerSentence:
    def setup_method(self):
        self.chunker = Chunker(strategy="sentence", chunk_size=50)

    def test_splits_on_sentence_boundary(self):
        chunker = Chunker(strategy="sentence", chunk_size=5, overlap=1)
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk(text, source="sentences.txt")

        assert len(chunks) >= 2
        assert all(c.text.strip().endswith(".") for c in chunks)
        rebuilt = " ".join(c.text for c in chunks)
        assert "First sentence." in rebuilt
        assert "Second sentence." in rebuilt
        assert "Third sentence." in rebuilt
