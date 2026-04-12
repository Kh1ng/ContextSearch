# embeddings.py
# Ollama embedding API wrapper and cosine similarity utility.
#
# Uses the local Ollama instance at localhost:11434.
# Default model: nomic-embed-text (384-dim, fast, good recall).

import math
import requests

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


def get_embedding(text: str, model: str = DEFAULT_EMBED_MODEL) -> list[float]:
    """
    Return a dense embedding vector for text via the local Ollama instance.
    Raises requests.HTTPError on API failure.
    """
    resp = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": model, "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity between two equal-length vectors.
    Returns a value in [-1, 1]; returns 0.0 if either vector is zero.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def embed_chunks(chunks, model: str = DEFAULT_EMBED_MODEL, verbose: bool = False) -> None:
    """
    Populate the .embedding field on each Chunk in-place.
    Calls Ollama once per chunk sequentially.
    """
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        if verbose and (i == 0 or (i + 1) % 10 == 0 or i + 1 == total):
            print(f"  Embedding chunks {i + 1}/{total}…", flush=True)
        chunk.embedding = get_embedding(chunk.text, model=model)
