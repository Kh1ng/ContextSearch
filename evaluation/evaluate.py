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
from pathlib import Path

from evaluation.load_scifact import download_scifact, load_scifact, corpus_to_chunks
from src.chunker import Chunker
from src.retriever import Retriever

BUDGETS = [512, 1024, 2048, 4096, 8192]
RESULTS_DIR = Path(__file__).parent.parent / "results"


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs found in the top-k retrieved."""
    raise NotImplementedError


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k retrieved docs that are relevant."""
    raise NotImplementedError


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalised discounted cumulative gain at k."""
    raise NotImplementedError


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Mean reciprocal rank of the first relevant result."""
    raise NotImplementedError


def run_evaluation(budget: int) -> dict:
    """
    Run A* retrieval on all SciFact test queries at the given token budget.
    Returns a dict of aggregated metrics.
    """
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="Evaluate ContextSearch on SciFact")
    parser.add_argument("--budget", type=int, default=None, help="Single token budget to test")
    args = parser.parse_args()

    budgets = [args.budget] if args.budget else BUDGETS
    RESULTS_DIR.mkdir(exist_ok=True)

    all_results = []
    for b in budgets:
        print(f"Running evaluation at budget={b}...")
        result = run_evaluation(b)
        all_results.append(result)
        print(f"  Recall@10: {result['recall_at_10']:.3f}  "
              f"Token savings: {result['token_savings']:.1%}")

    out_path = RESULTS_DIR / "eval_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
