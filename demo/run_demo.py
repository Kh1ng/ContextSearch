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
    python demo/run_demo.py --llm --query "..." --sweep-md-files --min-files 1 --max-files 10
"""

import sys
import time
import argparse
import textwrap
from datetime import datetime
from pathlib import Path

import json
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


def get_corpus_files(corpus_dir: Path) -> list[Path]:
    """Return sorted list of .md files in corpus_dir."""
    return sorted(corpus_dir.glob("*.md"))


def load_partial_corpus(retriever: Retriever, files: list[Path]) -> None:
    """Load only the given files into retriever, replacing any existing chunks."""
    chunks: list = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        chunks.extend(retriever.chunker.chunk(text, source=str(f)))
    retriever.load_chunks(chunks)


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

    return results, used_tokens, savings_pct


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

    return full_tok, full_time, opt_tok, opt_time


def run_single_benchmark(
    query: str,
    retriever: Retriever,
    files: list[Path],
    model: str,
    use_llm: bool,
) -> tuple[dict, dict]:
    """Run vanilla + enhanced benchmark for one query against files.
    Loads files into the retriever, runs query stats, optionally calls the LLM.
    Returns (vanilla_record, enhanced_record).
    """
    load_partial_corpus(retriever, files)
    num_chunks, total_tokens = load_corpus_stats(retriever)
    num_md_files = len(files)
    full_context = "\n\n---\n\n".join(f.read_text(encoding="utf-8") for f in files)

    results, _used_tokens, savings_pct = run_query_stats(retriever, query, total_tokens)

    if not results and retriever._chunks:
        print(
            f"  WARNING: retrieval returned 0 chunks for query \"{query}\" "
            f"with {num_md_files} file(s) loaded ({num_chunks} chunks available). "
            f"Query may contain no indexable keywords."
        )

    opt_ctx = "\n\n---\n\n".join(c.text for c in results)

    if use_llm:
        out("")
        full_tok, full_time, opt_tok, opt_time = run_llm_comparison(
            query, results, full_context, retriever._chunks, model
        )
        full_time = full_time if full_time > 0 else None
        opt_time = opt_time if opt_time > 0 else None
    else:
        full_tok = count_tokens(build_prompt(full_context, query))
        opt_tok = count_tokens(build_prompt(opt_ctx, query))
        full_time = None
        opt_time = None

    vanilla_rec = build_run_record(
        query=query, mode="vanilla",
        md_files_searched=num_md_files,
        total_chunks_available=num_chunks,
        chunks_sent=num_chunks,
        corpus_tokens=total_tokens,
        prompt_tokens_sent=full_tok,
        llm_latency_seconds=full_time,
        token_savings_percent=0,
        token_reduction_percent_vs_vanilla=0,
        latency_reduction_percent_vs_vanilla=0,
    )
    enhanced_rec = build_run_record(
        query=query, mode="enhanced",
        md_files_searched=num_md_files,
        total_chunks_available=num_chunks,
        chunks_sent=len(results),
        corpus_tokens=total_tokens,
        prompt_tokens_sent=opt_tok,
        llm_latency_seconds=opt_time,
        token_savings_percent=round(savings_pct, 1),
        token_reduction_percent_vs_vanilla=None,
        latency_reduction_percent_vs_vanilla=None,
    )
    return vanilla_rec, compute_comparison_metrics(vanilla_rec, enhanced_rec)


def run_sweep_benchmark(
    queries: list[str],
    all_files: list[Path],
    retriever: Retriever,
    model: str,
    use_llm: bool,
    min_files: int,
    max_files: int,
) -> list[dict]:
    """Benchmark across corpus sizes N=min_files..max_files. Returns list of run records."""
    max_n = min(max_files, len(all_files))
    runs: list[dict] = []
    for n in range(min_files, max_n + 1):
        files = all_files[:n]
        section(f"N={n} — {n} file{'s' if n != 1 else ''}")
        for query in queries:
            v_rec, e_rec = run_single_benchmark(query, retriever, files, model, use_llm)
            runs.extend([v_rec, e_rec])
    return runs


# ---------------------------------------------------------------------------
# JSON output helpers
# ---------------------------------------------------------------------------

def build_run_record(
    query: str,
    mode: str,
    md_files_searched: int,
    total_chunks_available: int,
    chunks_sent: int,
    corpus_tokens: int,
    prompt_tokens_sent,
    llm_latency_seconds,
    token_savings_percent: float,
    token_reduction_percent_vs_vanilla,
    latency_reduction_percent_vs_vanilla,
) -> dict:
    return {
        "query": query,
        "mode": mode,
        "md_files_searched": md_files_searched,
        "total_chunks_available": total_chunks_available,
        "chunks_sent": chunks_sent,
        "corpus_tokens": corpus_tokens,
        "prompt_tokens_sent": prompt_tokens_sent,
        "llm_latency_seconds": llm_latency_seconds,
        "token_savings_percent": token_savings_percent,
        "token_reduction_percent_vs_vanilla": token_reduction_percent_vs_vanilla,
        "latency_reduction_percent_vs_vanilla": latency_reduction_percent_vs_vanilla,
    }


def compute_comparison_metrics(vanilla: dict, enhanced: dict) -> dict:
    """Return a copy of *enhanced* with vs-vanilla comparison fields filled in."""
    v_tok = vanilla["prompt_tokens_sent"]
    e_tok = enhanced["prompt_tokens_sent"]
    v_lat = vanilla["llm_latency_seconds"]
    e_lat = enhanced["llm_latency_seconds"]
    result = dict(enhanced)
    result["token_reduction_percent_vs_vanilla"] = (
        round((1 - e_tok / v_tok) * 100, 1) if v_tok and e_tok else None
    )
    result["latency_reduction_percent_vs_vanilla"] = (
        round((1 - e_lat / v_lat) * 100, 1) if v_lat and e_lat else None
    )
    return result


def write_json_report(output_path, metadata: dict, runs: list) -> None:
    """Write a pretty-printed JSON benchmark report to *output_path*."""
    report = {**metadata, "runs": runs}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(json_output_path=None, report_timestamp=""):
    parser = argparse.ArgumentParser(description="ContextSearch real-world demo")
    parser.add_argument("--llm", action="store_true", help="Run LLM comparison via Ollama")
    parser.add_argument("--query", type=str, default=None, help="Run a single custom query")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="Token budget for ContextSearch")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--sweep-md-files", action="store_true", help="Sweep corpus sizes from --min-files to --max-files")
    parser.add_argument("--min-files", type=int, default=1, help="Minimum corpus size for sweep (default: 1)")
    parser.add_argument("--max-files", type=int, default=10, help="Maximum corpus size for sweep (default: 10)")
    parser.add_argument(
        "--heuristic",
        choices=["cosine", "keyword"],
        default="cosine",
        help="Relevance heuristic: cosine (default, uses nomic-embed-text) or keyword (overlap)",
    )
    args = parser.parse_args()

    # --- Load corpus ---
    header("ContextSearch — Real-World Demo")
    out(f"  Corpus    : {CORPUS_DIR}", f"- **Corpus**: `{CORPUS_DIR}`")
    out(f"  Budget    : {args.budget:,} tokens", f"- **Budget**: {args.budget:,} tokens")
    out(f"  Model     : {args.model}", f"- **Model**: `{args.model}`")
    out(f"  Heuristic : {args.heuristic}", f"- **Heuristic**: {args.heuristic}")

    all_files = get_corpus_files(CORPUS_DIR)
    num_md_files = len(all_files)

    retriever = Retriever(token_budget=args.budget, strategy="markdown", heuristic=args.heuristic)
    load_partial_corpus(retriever, all_files)
    num_chunks, total_tokens = load_corpus_stats(retriever)

    out(
        f"  Loaded {num_chunks} chunks from {num_md_files} files",
        f"- Loaded **{num_chunks} chunks** from {num_md_files} files",
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

    runs: list[dict] = []
    queries = [args.query] if args.query else DEMO_QUERIES

    if args.sweep_md_files:
        # --- Corpus-size sweep mode ---
        header("Corpus Size Sweep")
        out(
            f"  Sweeping N={args.min_files}..{min(args.max_files, num_md_files)} files × {len(queries)} {'query' if len(queries) == 1 else 'queries'}",
            f"- Sweeping **N={args.min_files}..{min(args.max_files, num_md_files)}** files × **{len(queries)}** {'query' if len(queries) == 1 else 'queries'}",
        )
        runs = run_sweep_benchmark(
            queries=queries,
            all_files=all_files,
            retriever=retriever,
            model=args.model,
            use_llm=args.llm,
            min_files=args.min_files,
            max_files=args.max_files,
        )
        n_sizes = min(args.max_files, num_md_files) - args.min_files + 1
        hr()
        section("Sweep Summary")
        out(
            f"  {n_sizes} corpus sizes × {len(queries)} {'query' if len(queries) == 1 else 'queries'} = {n_sizes * len(queries)} benchmarks ({len(runs)} run records)",
            f"- **{n_sizes}** corpus sizes × **{len(queries)}** {'query' if len(queries) == 1 else 'queries'} = **{n_sizes * len(queries)}** benchmarks",
        )
        hr()
        out("")
    else:
        # --- Single-run mode (original behavior) ---
        if args.llm:
            query = queries[0]
            header("Query")
            v_rec, e_rec = run_single_benchmark(query, retriever, all_files, args.model, True)
            runs.extend([v_rec, e_rec])

            if len(queries) > 1:
                header("Remaining Queries (token stats only)")
                for q in queries[1:]:
                    v_rec, e_rec = run_single_benchmark(q, retriever, all_files, args.model, False)
                    runs.extend([v_rec, e_rec])
        else:
            header("Token Savings by Query")
            for query in queries:
                v_rec, e_rec = run_single_benchmark(query, retriever, all_files, args.model, False)
                runs.extend([v_rec, e_rec])

        # Reload full corpus so the summary query loop is accurate
        load_partial_corpus(retriever, all_files)
        current_chunks, current_tokens = load_corpus_stats(retriever)

        hr()
        section("Summary")
        all_savings = []
        for q in queries:
            r = retriever.query(q)
            used = sum(c.token_count for c in r)
            all_savings.append((current_tokens - used) / current_tokens * 100)
        avg_savings = sum(all_savings) / len(all_savings)
        out(
            f"  Average token savings across {len(queries)} queries: {avg_savings:.1f}%",
            f"- **Average token savings** across {len(queries)} queries: **{avg_savings:.1f}%**",
        )
        out(
            f"  Full corpus: {current_tokens:,} tokens  avg ContextSearch: {current_tokens * (1 - avg_savings/100):,.0f} tokens",
            f"- Full corpus: **{current_tokens:,}** tokens → avg ContextSearch: **{current_tokens * (1 - avg_savings/100):,.0f}** tokens",
        )
        hr()
        out("")

    if json_output_path is not None:
        metadata = {
            "timestamp": report_timestamp,
            "corpus_dir": str(CORPUS_DIR),
            "model": args.model,
            "token_budget": args.budget,
            "total_corpus_tokens": total_tokens,
            "total_chunks": num_chunks,
            "total_md_files": num_md_files,
            "llm_enabled": args.llm,
        }
        try:
            write_json_report(json_output_path, metadata, runs)
            print(f"  JSON written: {json_output_path}")
        except Exception as exc:
            print(f"  ERROR writing JSON: {exc}")


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H%M%S")
    output_file = output_dir / f"{timestamp}.md"
    json_output_file = output_dir / f"{timestamp}.json"
    with output_file.open("w", encoding="utf-8") as _fh:
        _fh.write(f"<!-- Generated: {timestamp} -->\n\n")
        _md_out = _fh
        main(json_output_path=json_output_file, report_timestamp=timestamp)
    print(f"\nReport saved to : {output_file}")
    print(f"JSON saved to   : {json_output_file}")
