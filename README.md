# ContextSearch

A* search-based context window optimization for AI agent instruction files.

## Overview

Modern AI frameworks rely on markdown files (skills, agent instructions, CLAUDE.md, etc.) as context for LLM behavior. As these files grow in number, injecting all of them into every prompt becomes expensive and degrades performance. ContextSearch solves this by using A\* search to select only the chunks most relevant to the current task, staying within a configurable token budget.

### How it works

Given a query and a corpus of chunked documents, ContextSearch uses A\* to find the optimal subset of chunks where:

- **g(n)** = cumulative token cost of selected chunks
- **h(n)** = estimated relevance gain of remaining candidates (keyword overlap with query)
- **f(n) = g(n) + h(n)** guides selection toward maximum relevance within budget

## Project Structure

```
contextSearch/
├── src/
│   ├── chunker.py       # Splits documents into chunks by strategy
│   ├── tokenizer.py     # Token counting utilities
│   ├── search.py        # A* search algorithm
│   └── retriever.py     # Orchestrator: query → ranked chunks
├── evaluation/
│   ├── load_scifact.py  # Loads BEIR SciFact dataset
│   └── evaluate.py      # Computes accuracy and token savings
├── tests/
│   ├── test_chunker.py
│   ├── test_search.py
│   └── test_retriever.py
├── docs/
│   ├── design.md        # Architecture and algorithm design decisions
│   └── evaluation.md    # Evaluation methodology and metrics
└── requirements.txt
```

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd contextSearch

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from src.retriever import Retriever

retriever = Retriever(token_budget=2048)
retriever.load_corpus("path/to/docs/")

results = retriever.query("How do I configure authentication?")
for chunk in results:
    print(chunk.text)
```

## Evaluation

We evaluate against the [BEIR](https://github.com/beir-cellar/beir) SciFact subset, which provides pre-labeled query/document relevance pairs.

```bash
python evaluation/evaluate.py --budget 2048
```

See [docs/evaluation.md](docs/evaluation.md) for metrics and methodology.

## Team

- Colton Spurgin — chunking, tokenization, evaluation pipeline
- Michael Ramirez — A\* search, retriever orchestration, tests

_Much of the work was done collaboratively via live shared pair programming and git history is not a perfect reflection of individual contributions._

## License


