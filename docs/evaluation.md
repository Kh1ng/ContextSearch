# Evaluation

## Dataset: BEIR SciFact

We use the [SciFact](https://github.com/allenai/scifact) subset from the [BEIR benchmark](https://github.com/beir-cellar/beir). SciFact contains scientific claims paired with abstracts from biomedical literature, with binary relevance labels (supports / refutes / not enough info).

| Property | Value |
|----------|-------|
| Queries | 300 test queries |
| Corpus size | ~5,183 documents |
| Relevance labels | Binary (relevant / not relevant) |
| Domain | Biomedical / scientific claims |

We chose SciFact because:
- Pre-labeled relevance pairs allow objective accuracy measurement
- Small enough corpus to run full evaluation quickly
- Represents a realistic information retrieval task (not just keyword lookup)

### Download

```bash
# BEIR provides a pip package for easy dataset access
pip install beir
python evaluation/load_scifact.py  # downloads to ./data/scifact/
```

---

## Experimental Setup

### Baselines

| System | Description |
|--------|-------------|
| **Full context** | All corpus documents injected (no selection) — oracle upper bound on recall, worst case on token cost |
| **Random** | Chunks selected randomly up to budget — lower bound on accuracy |
| **ContextSearch (A\*)** | Our system at varying token budgets |

### Token budgets tested

We sweep over budgets to produce a cost vs. accuracy trade-off curve:

- 512 tokens
- 1024 tokens
- 2048 tokens
- 4096 tokens
- 8192 tokens

---

## Metrics

### Retrieval accuracy

We use standard IR metrics computed over the top-k selected chunks per query:

| Metric | Description |
|--------|-------------|
| **Recall@k** | Fraction of relevant documents retrieved in top k |
| **Precision@k** | Fraction of retrieved documents that are relevant |
| **NDCG@10** | Normalized discounted cumulative gain — rewards ranking relevant docs higher |
| **MRR** | Mean reciprocal rank of first relevant result |

Primary metric: **Recall@10** — for context window optimization, retrieving all relevant chunks matters more than ranking precision.

### Token efficiency

For each token budget B, we compare:

```
token_savings = (full_context_tokens - budget_tokens) / full_context_tokens
accuracy_retention = metric_A* / metric_full_context
```

The goal is to demonstrate that A\* can retain most accuracy (>0.85 accuracy retention) while significantly reducing token usage (>0.5 token savings).

---

## Running the Evaluation

```bash
# Full evaluation sweep across all budgets
python evaluation/evaluate.py

# Single budget
python evaluation/evaluate.py --budget 2048

# Output: results/eval_results.json and results/eval_summary.md
```

### Output format

```json
{
  "budget": 2048,
  "recall_at_10": 0.81,
  "precision_at_10": 0.43,
  "ndcg_at_10": 0.74,
  "mrr": 0.68,
  "avg_tokens_used": 1987,
  "token_savings": 0.61,
  "accuracy_retention": 0.89
}
```

---

## Limitations

- SciFact is a biomedical domain — keyword overlap heuristic may perform differently on code or agent instruction corpora (our real use case).
- Binary relevance labels don't capture partial relevance; a chunk that covers half of a claim still scores 0.
- We measure retrieval accuracy, not downstream LLM task performance — a chunk ranked #1 that the model can't use effectively is still counted as a hit.
