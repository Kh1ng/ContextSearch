# chunker.py
# Splits documents into chunks using configurable strategies.
# Owner: Colton Spurgin
#
# Strategies:
#   - fixed:    split every N tokens with optional overlap
#   - sentence: split on sentence boundaries
#   - markdown: split on markdown headings (default)
#
# TODO: implement Chunk dataclass and Chunker class

from dataclasses import dataclass, field
import re
from typing import Literal

from src.tokenizer import count_tokens, extract_keywords


ChunkStrategy = Literal["fixed", "sentence", "markdown"]


@dataclass
class Chunk:
    text: str
    token_count: int
    source: str               # originating file path
    index: int                # position in source document
    keywords: set[str] = field(default_factory=set)


class Chunker:
    def __init__(
        self,
        strategy: ChunkStrategy = "markdown",
        chunk_size: int = 256,
        overlap: int = 32,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        """Split text into Chunk objects using the configured strategy."""
        if not text or not text.strip():
            return []

        if self.strategy == "fixed":
            parts = self._chunk_fixed(text)
        elif self.strategy == "sentence":
            parts = self._chunk_sentence(text)
        elif self.strategy == "markdown":
            parts = self._chunk_markdown(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")

        chunks: list[Chunk] = []
        for idx, part in enumerate(parts):
            cleaned = part.strip()
            if not cleaned:
                continue
            chunks.append(
                Chunk(
                    text=cleaned,
                    token_count=count_tokens(cleaned),
                    source=source,
                    index=idx,
                    keywords=extract_keywords(cleaned),
                )
            )
        return chunks

    def _chunk_fixed(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []

        chunks: list[str] = []
        start = 0
        total_words = len(words)

        while start < total_words:
            end = min(start + self.chunk_size, total_words)
            chunk_text = " ".join(words[start:end]).strip()
            if chunk_text:
                chunks.append(chunk_text)

            if end >= total_words:
                break

            start = end - self.overlap

        return chunks

    def _chunk_sentence(self, text: str) -> list[str]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
        if not sentences:
            return []

        chunks: list[str] = []
        current: list[str] = []

        for sentence in sentences:
            sentence_tokens = count_tokens(sentence)

            if sentence_tokens > self.chunk_size:
                if current:
                    chunks.append(" ".join(current).strip())
                    current = []
                chunks.extend(self._chunk_fixed(sentence))
                continue

            candidate = (" ".join(current + [sentence])).strip()
            if current and count_tokens(candidate) > self.chunk_size:
                chunks.append(" ".join(current).strip())
                current = [sentence]
            else:
                current.append(sentence)

        if current:
            chunks.append(" ".join(current).strip())

        return chunks

    def _chunk_markdown(self, text: str) -> list[str]:
        lines = text.splitlines()
        if not lines:
            return []

        sections: list[str] = []
        current_lines: list[str] = []
        heading_pattern = re.compile(r"^\s{0,3}#{1,6}\s+\S")

        for line in lines:
            if heading_pattern.match(line) and current_lines:
                sections.append("\n".join(current_lines).strip())
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            sections.append("\n".join(current_lines).strip())

        chunks: list[str] = []
        for section in sections:
            if not section:
                continue
            if count_tokens(section) <= self.chunk_size:
                chunks.append(section)
            else:
                chunks.extend(self._chunk_fixed(section))

        return chunks
