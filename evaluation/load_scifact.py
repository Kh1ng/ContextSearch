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
import shutil
import zipfile
import tempfile
import sys
from pathlib import Path
from dataclasses import dataclass

import requests

# Allow running this file directly via: python evaluation/load_scifact.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunker import Chunk, Chunker

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
    required_paths = [
        data_dir / "corpus.jsonl",
        data_dir / "queries.jsonl",
        data_dir / "qrels" / "test.tsv",
    ]
    if all(p.exists() for p in required_paths):
        return

    data_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / "scifact.zip"
        with requests.get(SCIFACT_URL, stream=True, timeout=60) as response:
            response.raise_for_status()
            with zip_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)

        extracted_root = Path(tmp_dir) / "scifact"
        if not extracted_root.exists():
            raise FileNotFoundError("Downloaded SciFact archive did not contain expected 'scifact' folder")

        for item in extracted_root.iterdir():
            target = data_dir / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()

            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)


def load_scifact(data_dir: Path = DATA_DIR) -> SciFact:
    """
    Parse corpus.jsonl, queries.jsonl, and qrels/test.tsv from data_dir.
    Returns a SciFact dataclass.
    """
    corpus_path = data_dir / "corpus.jsonl"
    queries_path = data_dir / "queries.jsonl"
    qrels_path = data_dir / "qrels" / "test.tsv"

    if not corpus_path.exists() or not queries_path.exists() or not qrels_path.exists():
        missing = [str(p) for p in (corpus_path, queries_path, qrels_path) if not p.exists()]
        raise FileNotFoundError(
            "SciFact files are missing. Run download_scifact() first. Missing: " + ", ".join(missing)
        )

    corpus: dict[str, str] = {}
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = str(row.get("_id", "")).strip()
            if not doc_id:
                continue
            title = str(row.get("title", "")).strip()
            text = str(row.get("text", "")).strip()
            combined = f"{title}\n{text}".strip() if title else text
            corpus[doc_id] = combined

    queries: dict[str, str] = {}
    with queries_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row.get("_id", "")).strip()
            query_text = str(row.get("text", "")).strip()
            if query_id:
                queries[query_id] = query_text

    qrels: dict[str, set[str]] = {}
    with qrels_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            query_id = str(row.get("query-id", "")).strip()
            doc_id = str(row.get("corpus-id", "")).strip()
            score_raw = row.get("score", "0")

            try:
                score = int(score_raw)
            except (TypeError, ValueError):
                score = 0

            if score <= 0 or not query_id or not doc_id:
                continue

            qrels.setdefault(query_id, set()).add(doc_id)

    return SciFact(queries=queries, corpus=corpus, qrels=qrels)


def corpus_to_chunks(corpus: dict[str, str], chunker: Chunker) -> dict[str, list[Chunk]]:
    """
    Chunk every document in the corpus.
    Returns a mapping of doc_id -> list[Chunk].
    """
    chunked: dict[str, list[Chunk]] = {}
    for doc_id, text in corpus.items():
        chunked[doc_id] = chunker.chunk(text, source=doc_id)
    return chunked


def main() -> None:
    download_scifact()
    dataset = load_scifact()
    print(
        f"Loaded SciFact: {len(dataset.corpus)} docs, "
        f"{len(dataset.queries)} queries, {len(dataset.qrels)} qrel query entries"
    )


if __name__ == "__main__":
    main()
