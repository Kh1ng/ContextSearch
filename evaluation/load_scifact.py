# load_scifact.py
# Downloads and parses the BEIR SciFact dataset into Chunk objects.
# Owner: Colton Spurgin
#
# SciFact file layout after download:
#   data/scifact/corpus.jsonl   — {"_id", "title", "text", "metadata"}
#   data/scifact/queries.jsonl  — {"_id", "text", "metadata"}
#   data/scifact/qrels/test.tsv — query_id  corpus_id  score  (TSV, header row)
#
# Relevance scores in qrels: 1 = relevant, 0 = not relevant (we ignore 0 rows)

import json
import csv
from pathlib import Path
from dataclasses import dataclass

from src.chunker import Chunk, Chunker
from src.tokenizer import extract_keywords, count_tokens

DATA_DIR = Path(__file__).parent.parent / "data" / "scifact"
SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


@dataclass
class SciFact:
    """Container for a loaded SciFact split."""
    queries: dict[str, str]          # query_id -> query text
    corpus: dict[str, str]           # doc_id   -> document text (title + abstract)
    qrels: dict[str, set[str]]       # query_id -> set of relevant doc_ids


def download_scifact(data_dir: Path = DATA_DIR) -> None:
    """Download and unzip SciFact into data_dir if not already present."""
    raise NotImplementedError


def load_scifact(data_dir: Path = DATA_DIR) -> SciFact:
    """
    Parse corpus.jsonl, queries.jsonl, and qrels/test.tsv from data_dir.
    Returns a SciFact dataclass.
    """
    raise NotImplementedError


def corpus_to_chunks(corpus: dict[str, str], chunker: Chunker) -> dict[str, list[Chunk]]:
    """
    Chunk every document in the corpus.
    Returns a mapping of doc_id -> list[Chunk].
    """
    raise NotImplementedError
