#!/usr/bin/env python3
# demo_evaluation.py
# Shows actual queries and retrieved chunks for presentation demo.
# Usage: source .venv/bin/activate && python demo_evaluation.py --budget 512 --num-examples 5

import sys
from pathlib import Path
import argparse
import json

# Allow running this file directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.load_scifact import download_scifact, load_scifact, corpus_to_chunks
from src.chunker import Chunker
from src.retriever import Retriever
from src.tokenizer import count_tokens


def demo_evaluation(budget: int, num_examples: int = 5):
    """Run evaluation and show example queries with retrieved chunks."""
    print(f"\n{'='*80}")
    print(f"ContextSearch Demo Evaluation (Budget: {budget} tokens)")
    print(f"{'='*80}\n")
    
    # Load data
    print("Loading SciFact dataset...")
    download_scifact()
    dataset = load_scifact()
    print(f"✓ Loaded: {len(dataset.corpus)} docs, {len(dataset.queries)} queries, {len(dataset.qrels)} qrel entries\n")
    
    # Chunk corpus
    print("Chunking corpus with markdown strategy...")
    chunker = Chunker(strategy="markdown", chunk_size=256, overlap=32)
    all_chunks = corpus_to_chunks(dataset.corpus, chunker)
    total_chunks = sum(len(chunks) for chunks in all_chunks.values())
    print(f"✓ Created {total_chunks} chunks\n")
    
    # Build retriever
    flat_chunks = []
    for doc_id, chunks in all_chunks.items():
        flat_chunks.extend(chunks)
    
    retriever = Retriever(token_budget=budget)
    retriever.load_chunks(flat_chunks)
    
    # Find queries with relevance labels
    queries_with_qrels = [qid for qid in dataset.queries if qid in dataset.qrels]
    print(f"Found {len(queries_with_qrels)} queries with relevance labels\n")
    
    # Show examples
    print(f"{'='*80}")
    print(f"EXAMPLE QUERIES AND RETRIEVED CHUNKS")
    print(f"{'='*80}\n")
    
    example_count = min(num_examples, len(queries_with_qrels))
    
    for i, query_id in enumerate(queries_with_qrels[:example_count], 1):
        query_text = dataset.queries[query_id]
        relevant_doc_ids = dataset.qrels[query_id]
        
        # Run retrieval
        result_chunks = retriever.query(query_text)
        retrieved_doc_ids = [chunk.source for chunk in result_chunks]
        total_tokens = sum(chunk.token_count for chunk in result_chunks)
        
        # Check if any relevant docs found
        found_relevant = set(retrieved_doc_ids) & relevant_doc_ids
        recall = len(found_relevant) / len(relevant_doc_ids) if relevant_doc_ids else 0.0
        
        # Print query info
        print(f"\n[Example {i}] Query #{query_id}")
        print(f"-" * 80)
        print(f"Query: {query_text}")
        print(f"\nRelevant docs: {relevant_doc_ids}")
        print(f"Found relevant: {found_relevant if found_relevant else 'None ✗'}")
        print(f"Recall@k: {len(found_relevant)}/{len(relevant_doc_ids)} ({recall:.0%})")
        print(f"\nTokens used: {total_tokens} / {budget} (savings: {(1 - total_tokens/budget)*100:.1f}% of budget unused)")
        
        # Show retrieved chunks (FULL TEXT)
        print(f"\nRetrieved {len(result_chunks)} chunks:")
        for j, chunk in enumerate(result_chunks, 1):
            marker = "✓ RELEVANT" if chunk.source in relevant_doc_ids else "  "
            print(f"\n  [{j}] {marker} | tokens={chunk.token_count} | doc={chunk.source}")
            print(f"      {'─' * 76}")
            # Print full chunk text, wrapped at reasonable width
            for line in chunk.text.split('\n'):
                print(f"      {line}")
            print(f"      {'─' * 76}")
    
    print(f"\n{'='*80}")
    print("Demo complete! This shows why keyword heuristic struggles on SciFact:")
    print("- Queries use abstract terms (biomaterials, properties)")
    print("- Docs use different vocabulary (nanotechnologies, stem cells)")
    print("- No keyword overlap → chunks not selected even if semantically relevant")
    print("\nON AGENT INSTRUCTIONS: keywords are denser and more aligned!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo ContextSearch with example queries")
    parser.add_argument("--budget", type=int, default=512, help="Token budget")
    parser.add_argument("--num-examples", type=int, default=5, help="Number of example queries to show")
    args = parser.parse_args()
    
    demo_evaluation(args.budget, args.num_examples)
