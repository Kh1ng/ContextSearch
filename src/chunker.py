# chunker.py
# Splits documents into chunks using configurable strategies.
# Owner: Colton Spurgin
#
# Strategies:
#   - fixed:    split every N tokens with optional overlap
#   - sentence: split on sentence boundaries
#   - markdown: split on ## headings
#
# TODO: implement Chunk dataclass and Chunker class

from dataclasses import dataclass, field
from typing import Literal


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
        strategy: ChunkStrategy = "fixed",
        chunk_size: int = 256,
        overlap: int = 32,
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        """Split text into Chunk objects using the configured strategy."""
        raise NotImplementedError
