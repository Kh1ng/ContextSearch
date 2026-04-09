# retriever.py
# Orchestrates chunking, indexing, and A* search into a single query interface.
# Owner: Michael Ramirez
#
# Usage:
#   retriever = Retriever(token_budget=2048)
#   retriever.load_corpus("path/to/docs/")
#   chunks = retriever.query("How do I configure authentication?")

from pathlib import Path
from src.chunker import Chunk, Chunker, ChunkStrategy
from src.search import astar_search


class Retriever:
    def __init__(
        self,
        token_budget: int = 2048,
        strategy: ChunkStrategy = "markdown",
        chunk_size: int = 256,
        overlap: int = 32,
    ):
        self.token_budget = token_budget
        self.chunker = Chunker(strategy=strategy, chunk_size=chunk_size, overlap=overlap)
        self._chunks: list[Chunk] = []

    def load_corpus(self, corpus_path: str | Path) -> None:
        """
        Read all .md and .txt files under corpus_path, chunk them, and index.
        Populates self._chunks.
        """
        raise NotImplementedError

    def load_chunks(self, chunks: list[Chunk]) -> None:
        """Directly load pre-built chunks (used by the evaluation harness)."""
        self._chunks = chunks

    def query(self, query: str) -> list[Chunk]:
        """
        Run A* over indexed chunks and return selected chunks within token budget.
        """
        if not self._chunks:
            raise RuntimeError("Corpus not loaded. Call load_corpus() or load_chunks() first.")
        return astar_search(query, self._chunks, self.token_budget)
