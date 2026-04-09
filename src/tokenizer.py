# tokenizer.py
# Token counting utilities using tiktoken (cl100k_base encoding).
# Owner: Colton Spurgin
#
# Responsibilities:
#   - count tokens in a string
#   - strip stopwords and punctuation to produce a keyword set for heuristic use

import re
import tiktoken

# Shared encoder instance — cl100k_base is used by GPT-4 and is a reasonable
# approximation for Claude's tokenizer.
_ENCODER = tiktoken.get_encoding("cl100k_base")

# Basic English stopwords — extend as needed
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "that", "this", "these", "those",
    "it", "its", "as", "not", "no", "if", "so", "than", "then", "when",
}


def count_tokens(text: str) -> int:
    """Return the number of tokens in text."""
    return len(_ENCODER.encode(text))


def extract_keywords(text: str) -> set[str]:
    """Return a set of lowercase, non-stopword words from text."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if w not in _STOPWORDS}
