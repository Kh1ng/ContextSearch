"""
ContextSearch Real-World Demo
==============================
Loads a corpus of agent instruction markdown files, runs A* retrieval
for several realistic queries, and optionally calls a local Ollama instance
to show side-by-side LLM responses (full context vs optimized context).

Usage:
    python demo/run_demo.py                         # token stats only
    python demo/run_demo.py --llm                   # include Ollama responses
    python demo/run_demo.py --llm --query "..."     # single custom query with LLM
    python demo/run_demo.py --model mistral         # different Ollama model
"""

import sys
import time
import argparse
import textwrap
from datetime import datetime
from pathlib import Path

import requests

# Allow running from the project root or from demo/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import Retriever
from src.tokenizer import count_tokens

# ---------------------------------------------------------------------------
# Dual output — plain text to terminal, markdown to report file
# ---------------------------------------------------------------------------

_md_out = None  # file handle assigned by __main__

def out(plain: str, md: str | None = None) -> None:
    """Print plain text to the terminal; write markdown to the report file."""
    print(plain)
    if _md_out is not None:
        _md_out.write((md if md is not None else plain) + "\n")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CORPUS_DIR = Path(__file__).parent / "corpus"
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:14b-instruct"
DEFAULT_BUDGET = 1024  # tokens fed to the LLM from ContextSearch

DEMO_QUERIES = [
    "How do I set up JWT authentication and configure token expiry?",
    "What is the process for running and rolling back database migrations?",
    "How do I deploy to production and what approvals are required?",
    "What are the testing conventions for writing new API endpoint tests?",
    "How should I format git commit messages and structure a pull request?",
]

# Queries designed to surface easter eggs — great for live demos
EASTER_EGG_QUERIES = [
    "Is there anything special I need to do before running a database migration?",
    "What is the process for deploying on a Friday?",
    "What do I need to include in a pull request that touches the billing module?",
    "What comment is required at the top of every test file?",
]

SYSTEM_PROMPT = (
    "You are a helpful engineering assistant. "
    "Answer the question using only the provided context. "
    "Be concise — aim for 3-5 sentences."
)

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

WIDTH = 72

def hr():
    out("─" * WIDTH, "\n---\n")

def header(title: str):
    t = title.strip()
    out(f"\n{'═' * WIDTH}\n  {t}\n{'═' * WIDTH}", f"\n# {t}\n")

def section(title: str):
    t = title.strip()
    out(f"\n  {t}\n  {'─' * (WIDTH - 4)}", f"\n### {t}\n")

def wrap(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=WIDTH - indent, initial_indent=prefix, subsequent_indent=prefix)

def token_bar(used: int, total: int, width: int = 30) -> str:
    filled = int((used / total) * width) if total else 0
    return "[" + "█" * filled + "░" * (width - filled) + "]"

# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def check_ollama(model: str) -> bool:
    """Return True if Ollama is reachable and the model is available."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return any(m == model or m.startswith(model.split(":")[0]) for m in models)
    except Exception:
        return False


def call_ollama(prompt: str, model: str) -> tuple[str, float]:
    """Call the Ollama generate endpoint. Returns (response_text, elapsed_seconds)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 256,
        },
    }
    t0 = time.time()
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    elapsed = time.time() - t0
    return resp.json().get("response", "").strip(), elapsed


def build_prompt(context: str, query: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"

# ---------------------------------------------------------------------------
# Demo logic
# ---------------------------------------------------------------------------

def load_corpus_stats(retriever: Retriever) -> tuple[int, int]:
    """Return (total_chunks, total_tokens) for the loaded corpus."""
    total_tokens = sum(c.token_count for c in retriever._chunks)
    return len(retriever._chunks), total_tokens


def run_query_stats(retriever: Retriever, query: str, full_token_count: int):
    """Print token comparison for one query. Returns selected chunks."""
    all_chunks = retriever._chunks
    results = retriever.query(query)
    used_tokens = sum(c.token_count for c in results)
    savings_pct = (1 - used_tokens / full_token_count) * 100 if full_token_count else 0

    out(f'\n  Query: "{query}"', f'\n**Query:** "{query}"\n')

    # --- Full context side ---
    bar = token_bar(full_token_count, full_token_count)
    out(
        f"    FULL CONTEXT — {full_token_count:,} tokens, {len(all_chunks)} chunks  {bar}",
        f"**FULL CONTEXT** — {full_token_count:,} tokens, {len(all_chunks)} chunks  {bar}\n",
    )

    # Summarise by file so it doesn't scroll forever
    file_stats: dict[str, tuple[int, int]] = {}  # filename -> (chunk_count, token_sum)
    for chunk in all_chunks:
        fname = Path(chunk.source).name
        c, t = file_stats.get(fname, (0, 0))
        file_stats[fname] = (c + 1, t + chunk.token_count)
    for fname, (count, tokens) in sorted(file_stats.items()):
        out(
            f"      [{fname}]  {count} chunks  {tokens:,} tok",
            f"- `{fname}` — {count} chunks, {tokens:,} tokens",
        )

    out("")

    # --- ContextSearch side ---
    bar2 = token_bar(used_tokens, full_token_count)
    out(
        f"    CONTEXTSEARCH — {used_tokens:,} tokens, {len(results)} chunks  {bar2}  ({savings_pct:.1f}% saved)",
        f"**CONTEXTSEARCH** — {used_tokens:,} tokens, {len(results)} chunks  {bar2}  ({savings_pct:.1f}% saved)\n",
    )
    for chunk in results:
        src = Path(chunk.source).name
        preview = chunk.text.replace("\n", " ")
        out(
            f"      [{src}]  {chunk.token_count:>4} tok  \"{preview}\"",
            f"- `{src}` — {chunk.token_count:>4} tok — \"{preview}\"",
        )

    return results


def run_llm_comparison(query: str, results, full_context: str, all_chunks: list, model: str):
    """Call Ollama with full context and optimized context, print both responses."""
    optimized_context = "\n\n---\n\n".join(c.text for c in results)

    full_prompt = build_prompt(full_context, query)
    opt_prompt = build_prompt(optimized_context, query)
    full_tok = count_tokens(full_prompt)
    opt_tok = count_tokens(opt_prompt)

    section(f"FULL CONTEXT → LLM  ({len(all_chunks)} chunks, {full_tok:,} tokens sent)")
    out(f"    Calling {model}…", f"Calling `{model}`…\n")
    try:
        full_response, full_time = call_ollama(full_prompt, model)
        out(f"    Time: {full_time:.1f}s\n", f"**Time:** {full_time:.1f}s\n")
        out(wrap(full_response), full_response)
    except Exception as e:
        out(f"    ERROR: {e}", f"> **ERROR:** {e}")
        full_time = 0

    section(f"CONTEXTSEARCH → LLM  ({len(results)} chunks, {opt_tok:,} tokens sent)")
    out(f"    Calling {model}…", f"Calling `{model}`…\n")
    try:
        opt_response, opt_time = call_ollama(opt_prompt, model)
        out(f"    Time: {opt_time:.1f}s\n", f"**Time:** {opt_time:.1f}s\n")
        out(wrap(opt_response), opt_response)
    except Exception as e:
        out(f"    ERROR: {e}", f"> **ERROR:** {e}")
        opt_time = 0

    if full_time and opt_time:
        section("COMPARISON SUMMARY")
        out(
            f"    Chunks sent     : {len(all_chunks)} → {len(results)}  ({len(all_chunks) - len(results)} dropped)",
            f"- **Chunks sent**: {len(all_chunks)} → {len(results)}  ({len(all_chunks) - len(results)} dropped)",
        )
        out(
            f"    Token reduction : {full_tok:,} → {opt_tok:,}  ({(1 - opt_tok/full_tok)*100:.1f}% fewer)",
            f"- **Token reduction**: {full_tok:,} → {opt_tok:,}  ({(1 - opt_tok/full_tok)*100:.1f}% fewer)",
        )
        out(
            f"    Latency         : {full_time:.1f}s → {opt_time:.1f}s  ({full_time/opt_time:.1f}× faster, {(1 - opt_time/full_time)*100:.1f}% less)",
            f"- **Latency**: {full_time:.1f}s → {opt_time:.1f}s  ({full_time/opt_time:.1f}× faster, {(1 - opt_time/full_time)*100:.1f}% less)",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ContextSearch real-world demo")
    parser.add_argument("--llm", action="store_true", help="Run LLM comparison via Ollama")
    parser.add_argument("--query", type=str, default=None, help="Run a single custom query")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="Token budget for ContextSearch")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model name")
    args = parser.parse_args()

    # --- Load corpus ---
    header("ContextSearch — Real-World Demo")
    out(f"  Corpus : {CORPUS_DIR}", f"- **Corpus**: `{CORPUS_DIR}`")
    out(f"  Budget : {args.budget:,} tokens", f"- **Budget**: {args.budget:,} tokens")
    out(f"  Model  : {args.model}", f"- **Model**: `{args.model}`")

    retriever = Retriever(token_budget=args.budget, strategy="markdown")
    retriever.load_corpus(CORPUS_DIR)
    num_chunks, total_tokens = load_corpus_stats(retriever)

    out(
        f"  Loaded {num_chunks} chunks from {len(list(CORPUS_DIR.glob('*.md')))} files",
        f"- Loaded **{num_chunks} chunks** from {len(list(CORPUS_DIR.glob('*.md')))} files",
    )
    out(f"  Total corpus tokens: {total_tokens:,}", f"- Total corpus tokens: **{total_tokens:,}**")

    # --- Check Ollama if needed ---
    if args.llm:
        out("")
        if check_ollama(args.model):
            out(f"  Ollama OK  ({args.model} available)", f"- Ollama ✓  (`{args.model}` available)")
        else:
            out(
                f"  WARNING: Ollama not reachable or model '{args.model}' not found.",
                f"> **WARNING**: Ollama not reachable or model `{args.model}` not found.",
            )
            out(f"  Run:  ollama pull {args.model}", f"> Run: `ollama pull {args.model}`")
            out("  Falling back to token stats only.", "> Falling back to token stats only.")
            args.llm = False

    # --- Build full context string (for LLM baseline) ---
    full_context = "\n\n---\n\n".join(
        f.read_text(encoding="utf-8") for f in sorted(CORPUS_DIR.glob("*.md"))
    )

    # --- Run queries ---
    queries = [args.query] if args.query else DEMO_QUERIES

    if args.llm:
        # LLM mode: one query at a time with full comparison
        query = queries[0]
        header("Query")
        results = run_query_stats(retriever, query, total_tokens)
        out("")
        run_llm_comparison(query, results, full_context, retriever._chunks, args.model)

        if len(queries) > 1:
            header("Remaining Queries (token stats only)")
            for q in queries[1:]:
                run_query_stats(retriever, q, total_tokens)
    else:
        # Stats-only mode: all queries
        header("Token Savings by Query")
        for query in queries:
            run_query_stats(retriever, query, total_tokens)

    # --- Final summary ---
    hr()
    section("Summary")
    all_savings = []
    for q in queries:
        r = retriever.query(q)
        used = sum(c.token_count for c in r)
        all_savings.append((total_tokens - used) / total_tokens * 100)
    avg_savings = sum(all_savings) / len(all_savings)
    out(
        f"  Average token savings across {len(queries)} queries: {avg_savings:.1f}%",
        f"- **Average token savings** across {len(queries)} queries: **{avg_savings:.1f}%**",
    )
    out(
        f"  Full corpus: {total_tokens:,} tokens  avg ContextSearch: {total_tokens * (1 - avg_savings/100):,.0f} tokens",
        f"- Full corpus: **{total_tokens:,}** tokens → avg ContextSearch: **{total_tokens * (1 - avg_savings/100):,.0f}** tokens",
    )
    hr()
    out("")


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H%M%S")
    output_file = output_dir / f"{timestamp}.md"
    with output_file.open("w", encoding="utf-8") as _fh:
        _fh.write(f"<!-- Generated: {timestamp} -->\n\n")
        _md_out = _fh
        main()
    print(f"\nOutput saved to: {output_file}")
