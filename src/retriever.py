# retriever.py
# Orchestrates chunking, indexing, and A* search into a single query interface.
# Owner: Michael Ramirez
#
# Usage:
#   retriever = Retriever(token_budget=2048)                        # cosine (default)
#   retriever = Retriever(token_budget=2048, heuristic="keyword")   # keyword overlap
#   retriever.load_corpus("path/to/docs/")
#   chunks = retriever.query("How do I configure authentication?")

from pathlib import Path
from typing import Literal

from src.chunker import Chunk, Chunker, ChunkStrategy
from src.search import astar_search, _keyword_overlap
from src.tokenizer import extract_keywords

Heuristic = Literal["cosine", "keyword"]


class Retriever:
    def __init__(
        self,
        token_budget: int = 2048,
        strategy: ChunkStrategy = "markdown",
        chunk_size: int = 256,
        overlap: int = 32,
        heuristic: Heuristic = "cosine",
        embed_model: str = "nomic-embed-text",
    ):
        self.token_budget = token_budget
        self.chunker = Chunker(strategy=strategy, chunk_size=chunk_size, overlap=overlap)
        self.heuristic = heuristic
        self.embed_model = embed_model
        self._chunks: list[Chunk] = []

    def load_corpus(self, corpus_path: str | Path) -> None:
        """
        Read all .md and .txt files under corpus_path, chunk them, and index.
        Populates self._chunks.  Computes embeddings if heuristic="cosine".
        """
        path = Path(corpus_path)

        if path.is_dir():
            files = list(path.glob("**/*.md")) + list(path.glob("**/*.txt"))
        else:
            files = [path]

        for file_path in files:
            text = file_path.read_text(encoding="utf-8")
            chunks = self.chunker.chunk(text, source=str(file_path))
            self._chunks.extend(chunks)

        if self.heuristic == "cosine":
            self._build_embeddings(verbose=True)

    def load_chunks(self, chunks: list[Chunk]) -> None:
        """
        Directly load pre-built chunks (used by the evaluation harness).
        Computes embeddings if heuristic="cosine" and they are not already set.
        """
        self._chunks = chunks
        if self.heuristic == "cosine" and any(c.embedding is None for c in self._chunks):
            self._build_embeddings(verbose=False)

    def _build_embeddings(self, verbose: bool = False) -> None:
        """Populate .embedding on all chunks via Ollama."""
        from src.embeddings import embed_chunks
        embed_chunks(self._chunks, model=self.embed_model, verbose=verbose)

    def query(self, query: str) -> list[Chunk]:
        """
        Run A* over indexed chunks and return selected chunks within token budget.
        """
        if not self._chunks:
            raise RuntimeError("Corpus not loaded. Call load_corpus() or load_chunks() first.")

        if self.heuristic == "cosine":
            from src.embeddings import get_embedding, cosine_similarity
            query_emb = get_embedding(query, model=self.embed_model)
            heuristic_fn = lambda chunk: (
                cosine_similarity(chunk.embedding, query_emb)
                if chunk.embedding is not None
                else 0.0
            )
        else:
            query_keywords = extract_keywords(query)
            heuristic_fn = lambda chunk: _keyword_overlap(chunk.keywords, query_keywords)

        return astar_search(query, self._chunks, self.token_budget, heuristic_fn)
