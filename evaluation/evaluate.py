# evaluate.py
# Runs the full evaluation sweep over SciFact and reports IR metrics.
# Owner: Colton Spurgin
#
# Metrics computed per token budget:
#   - Recall@10, Precision@10, NDCG@10, MRR
#   - Average tokens used, token savings vs full-context, accuracy retention
#
# Usage:
#   python evaluation/evaluate.py                  # all budgets
#   python evaluation/evaluate.py --budget 2048    # single budget

import argparse
import json
import sys
import math
from pathlib import Path
from collections import defaultdict

# Allow running this file directly via: python evaluation/evaluate.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests

from evaluation.load_scifact import download_scifact, load_scifact, corpus_to_chunks
from src.chunker import Chunker
from src.retriever import Retriever
from src.tokenizer import count_tokens

BUDGETS = [512, 1024, 2048, 4096, 8192]
RESULTS_DIR = Path(__file__).parent.parent / "results"


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs found in the top-k retrieved."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    found = len(set(top_k) & relevant)
    return found / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k retrieved docs that are relevant."""
    if k == 0 or not retrieved:
        return 0.0
    top_k = retrieved[:k]
    found = len(set(top_k) & relevant)
    return found / len(top_k)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalised discounted cumulative gain at k."""
    if not relevant:
        return 0.0

    top_k = retrieved[:k]
    dcg = 0.0
    for i, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 1)

    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))

    return dcg / idcg if idcg > 0 else 0.0


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Mean reciprocal rank of the first relevant result."""
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def run_evaluation(retriever: Retriever, budget: int, dataset, total_corpus_tokens: int) -> dict:
    """
    Run A* retrieval on all SciFact test queries at the given token budget.
    Reuses a pre-loaded (and pre-embedded) retriever — only token_budget changes.
    Returns a dict of aggregated metrics.
    """
    retriever.token_budget = budget

    recall_at_10_list = []
    precision_at_10_list = []
    ndcg_at_10_list = []
    mrr_list = []
    tokens_used_list = []

    for query_id, query_text in dataset.queries.items():
        if query_id not in dataset.qrels:
            continue

        relevant_doc_ids = dataset.qrels[query_id]

        result_chunks = retriever.query(query_text)
        retrieved_doc_ids = [chunk.source for chunk in result_chunks]
        tokens_used = sum(chunk.token_count for chunk in result_chunks)

        recall_at_10_list.append(recall_at_k(retrieved_doc_ids, relevant_doc_ids, 10))
        precision_at_10_list.append(precision_at_k(retrieved_doc_ids, relevant_doc_ids, 10))
        ndcg_at_10_list.append(ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, 10))
        mrr_list.append(mrr(retrieved_doc_ids, relevant_doc_ids))
        tokens_used_list.append(tokens_used)

    avg_recall = sum(recall_at_10_list) / len(recall_at_10_list) if recall_at_10_list else 0.0
    avg_precision = sum(precision_at_10_list) / len(precision_at_10_list) if precision_at_10_list else 0.0
    avg_ndcg = sum(ndcg_at_10_list) / len(ndcg_at_10_list) if ndcg_at_10_list else 0.0
    avg_mrr = sum(mrr_list) / len(mrr_list) if mrr_list else 0.0
    avg_tokens_used = sum(tokens_used_list) / len(tokens_used_list) if tokens_used_list else 0.0

    token_savings = (total_corpus_tokens - avg_tokens_used) / total_corpus_tokens if total_corpus_tokens > 0 else 0.0

    return {
        "budget": budget,
        "recall_at_10": avg_recall,
        "precision_at_10": avg_precision,
        "ndcg_at_10": avg_ndcg,
        "mrr": avg_mrr,
        "avg_tokens_used": avg_tokens_used,
        "token_savings": token_savings,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ContextSearch on SciFact")
    parser.add_argument("--budget", type=int, default=None, help="Single token budget to test")
    parser.add_argument(
        "--heuristic",
        choices=["cosine", "keyword"],
        default="cosine",
        help="Relevance heuristic (default: cosine)",
    )
    parser.add_argument("--out", type=str, default=None, help="Output JSON filename (default: eval_results.json)")
    args = parser.parse_args()

    budgets = [args.budget] if args.budget else BUDGETS
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load corpus and build embeddings ONCE — reused across all budget levels
    print("Loading SciFact corpus...")
    download_scifact()
    dataset = load_scifact()

    chunker = Chunker(strategy="markdown", chunk_size=256, overlap=32)
    all_chunks = corpus_to_chunks(dataset.corpus, chunker)
    flat_chunks = [c for chunks in all_chunks.values() for c in chunks]

    total_corpus_tokens = sum(count_tokens(text) for text in dataset.corpus.values())

    print(f"Corpus: {len(dataset.corpus)} docs, {len(flat_chunks)} chunks, {total_corpus_tokens:,} tokens")
    print(f"Heuristic: {args.heuristic}")

    retriever = Retriever(token_budget=budgets[0], heuristic=args.heuristic)
    print("Building index (embeddings if cosine)...")
    retriever.load_chunks(flat_chunks)
    print("Index ready.")

    all_results = []
    for b in budgets:
        print(f"Running evaluation at budget={b}...")
        result = run_evaluation(retriever, b, dataset, total_corpus_tokens)
        all_results.append(result)
        print(f"  Recall@10: {result['recall_at_10']:.3f}  Token savings: {result['token_savings']:.1%}")

    out_filename = args.out or ("eval_results_cosine.json" if args.heuristic == "cosine" else "eval_results.json")
    out_path = RESULTS_DIR / out_filename
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
