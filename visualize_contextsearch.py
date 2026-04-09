#!/usr/bin/env python3
"""
visualize_contextsearch.py

Dataset-level visualisation tool for ContextSearch demo-log markdown files.

Each .md file represents ONE experiment run (one query).  This script reads
every .md file in a folder, parses each into one row, and generates plots
that compare the distribution of runs across the whole dataset.

Outputs (all written to --output-dir, default: plots/)
  contextsearch_metrics.csv      — one row per markdown file
  scatter_full_context.png       — FC tokens sent vs FC latency
  scatter_contextsearch.png      — CS tokens sent vs CS latency
  scatter_comparison.png         — both methods on one axes, pairs connected
  bar_token_reduction.png        — token reduction % per file
  bar_latency_reduction.png      — latency reduction % per file

CLI
  python visualize_contextsearch.py ./demo_logs/
  python visualize_contextsearch.py ./demo_logs/ --output-dir results/plots
  python visualize_contextsearch.py demo.md          # single file also accepted
"""

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Regex building-blocks
# ─────────────────────────────────────────────────────────────────────────────

# Right-arrow: Unicode → (U+2192) or plain ASCII > or ->
_ARROW = r"(?:→|->|>)+"

# Integer that may contain comma thousands-separators, e.g. "4,506"
_NUM = r"[\d,]+"


# ─────────────────────────────────────────────────────────────────────────────
# Numeric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _int(s: str) -> int:
    """Convert a possibly-comma-separated integer string to int."""
    return int(s.replace(",", ""))


def _float(s: str) -> float:
    return float(s)


# ─────────────────────────────────────────────────────────────────────────────
# Core text parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_demo_text(text: str) -> Optional[dict]:
    """
    Extract all metrics from the raw markdown of one ContextSearch demo log.

    Returns a dict on success, or None if required fields cannot be found
    (so the caller can skip and warn about the file).

    Key design notes
    ────────────────
    • Numbers may contain commas ("4,410") — _int() handles them.

    • The markdown uses Unicode em-dash — (U+2014) between the method name
      and the token count in banner lines.  We match it with \\S+ (any non-
      space) so minor formatting variation is tolerated.

    • "full_context_tokens" / "contextsearch_tokens" come from the banner
      lines that appear BEFORE the LLM is called.  These reflect the context
      window budget.

    • "tokens_sent_full" / "tokens_sent_contextsearch" come from the
      ### ... → LLM section headers, e.g.
          ### FULL CONTEXT → LLM  (76 chunks, 4,506 tokens sent)
      These may be slightly larger than the banner values because the actual
      prompt includes system-prompt overhead.  The COMPARISON SUMMARY
      likewise uses the sent-token counts.

    • Times are extracted from within each ### section body with re.DOTALL,
      not from the COMPARISON SUMMARY, so they are always the raw measured
      values.

    • latency_reduction_percent and token_reduction_percent are taken from the
      COMPARISON SUMMARY when present; otherwise derived from measured values.
    """
    record: dict = {}

    # ── Query text ────────────────────────────────────────────────────────────
    # **Query:** "What do I need to do …?"
    m = re.search(r'\*\*Query:\*\*\s+"(.+?)"', text)
    record["query"] = m.group(1).strip() if m else ""

    # ── Corpus overview ───────────────────────────────────────────────────────
    # - Loaded **76 chunks** from 7 files
    m = re.search(
        r"Loaded\s+\*\*(" + _NUM + r")\s+chunks\*\*\s+from\s+(" + _NUM + r")\s+files",
        text,
    )
    if not m:
        return None
    record["loaded_chunks"] = _int(m.group(1))
    record["corpus_files"]  = _int(m.group(2))

    # - Total corpus tokens: **4,410**
    m = re.search(r"Total corpus tokens:\s+\*\*(" + _NUM + r")\*\*", text)
    if not m:
        return None
    record["total_corpus_tokens"] = _int(m.group(1))

    # ── FULL CONTEXT banner ───────────────────────────────────────────────────
    # **FULL CONTEXT** — 4,410 tokens, 76 chunks  [████…]
    # The dash between the method name and the token count might be an em-dash
    # (—), an en-dash (–), or hyphens; \S+ consumes whatever is there.
    m = re.search(
        r"\*\*FULL CONTEXT\*\*\s+\S+\s+(" + _NUM + r")\s+tokens,\s+(" + _NUM + r")\s+chunks",
        text,
    )
    if not m:
        return None
    record["full_context_tokens"] = _int(m.group(1))
    record["full_context_chunks"] = _int(m.group(2))

    # ── CONTEXTSEARCH banner ──────────────────────────────────────────────────
    # **CONTEXTSEARCH** — 968 tokens, 14 chunks  [██░░…]  (78.0% saved)
    # We also capture the "% saved" figure as a cross-check but do NOT use it
    # as the primary token_reduction_percent (that comes from COMPARISON SUMMARY).
    m = re.search(
        r"\*\*CONTEXTSEARCH\*\*\s+\S+\s+(" + _NUM + r")\s+tokens,\s+(" + _NUM + r")\s+chunks"
        r".*?\(([\d.]+)%\s+saved\)",
        text,
        re.DOTALL,
    )
    if not m:
        return None
    record["contextsearch_tokens"] = _int(m.group(1))
    record["contextsearch_chunks"] = _int(m.group(2))
    # _banner_pct_saved = _float(m.group(3))   # available if needed later

    # ── Section-scoped time extraction ───────────────────────────────────────
    #
    # We find each ### LLM-call section and extract **Time:** from its body.
    # This prevents accidentally matching a time from the wrong section.
    #
    # Section format:
    #   ### FULL CONTEXT → LLM  (76 chunks, 4,506 tokens sent)
    #   Calling `model`…
    #   **Time:** 57.6s
    #   <response text…>
    #
    # The section ends at the next ### or at end-of-string (\Z).

    def _extract_section(header_re: str) -> tuple[Optional[int], Optional[float]]:
        """
        Return (tokens_sent, time_seconds) from a matching ### section.
        Either value may be None if not found.
        """
        pat = re.compile(
            header_re
            + r"(?:\s+\((" + _NUM + r")\s+chunks,\s+(" + _NUM + r")\s+tokens\s+sent\))?"
            + r"(.*?)(?=###|\Z)",
            re.DOTALL | re.IGNORECASE,
        )
        sec = pat.search(text)
        if not sec:
            return None, None
        tokens = _int(sec.group(2)) if sec.group(2) else None
        t = re.search(r"\*\*Time:\*\*\s+([\d.]+)s", sec.group(3))
        time_s = _float(t.group(1)) if t else None
        return tokens, time_s

    fc_tokens_sent, fc_time = _extract_section(
        r"###\s+FULL\s+CONTEXT\s+" + _ARROW + r"\s+LLM"
    )
    cs_tokens_sent, cs_time = _extract_section(
        r"###\s+CONTEXTSEARCH\s+" + _ARROW + r"\s+LLM"
    )

    # Fall back to banner token counts if section headers were not parsed
    record["tokens_sent_full"]          = fc_tokens_sent if fc_tokens_sent is not None else record["full_context_tokens"]
    record["full_time_seconds"]         = fc_time

    record["tokens_sent_contextsearch"] = cs_tokens_sent if cs_tokens_sent is not None else record["contextsearch_tokens"]
    record["contextsearch_time_seconds"] = cs_time

    # ── COMPARISON SUMMARY ────────────────────────────────────────────────────

    # - **Chunks sent**: 76 → 14  (62 dropped)
    m = re.search(
        r"\*\*Chunks sent\*\*:\s+" + _NUM + r"\s+" + _ARROW + r"\s+" + _NUM
        + r"\s+\((" + _NUM + r")\s+dropped\)",
        text,
    )
    if m:
        record["chunks_dropped"] = _int(m.group(1))
    else:
        # Derive from banner chunk counts
        record["chunks_dropped"] = (
            record["full_context_chunks"] - record["contextsearch_chunks"]
        )

    # - **Token reduction**: 4,506 → 1,037  (77.0% fewer)
    # NOTE: these numbers are tokens *sent to the LLM* (with prompt overhead),
    # NOT the raw banner token counts.  Do not confuse them.
    m = re.search(
        r"\*\*Token reduction\*\*:\s+(" + _NUM + r")\s+" + _ARROW
        + r"\s+(" + _NUM + r")\s+\(([\d.]+)%\s+fewer\)",
        text,
    )
    if m:
        record["tokens_saved_absolute"]   = _int(m.group(1)) - _int(m.group(2))
        record["token_reduction_percent"] = _float(m.group(3))
    else:
        # Derive from measured sent-token counts
        tf = record["tokens_sent_full"]
        tc = record["tokens_sent_contextsearch"]
        if tf and tc and tf > 0:
            record["tokens_saved_absolute"]   = tf - tc
            record["token_reduction_percent"] = round(100.0 * (tf - tc) / tf, 1)
        else:
            record["tokens_saved_absolute"]   = None
            record["token_reduction_percent"] = None

    # - **Latency**: 57.6s → 13.9s  (4.1× faster, 75.8% less)
    # Accepts × (U+00D7) or plain x/X as the multiplication symbol.
    m = re.search(
        r"\*\*Latency\*\*:.*?\(([\d.]+)\s*[×xX]\s*faster,\s*([\d.]+)%\s+less\)",
        text,
    )
    if m:
        record["latency_reduction_percent"] = _float(m.group(2))
    else:
        # Derive from measured times
        ft = record.get("full_time_seconds")
        ct = record.get("contextsearch_time_seconds")
        if ft and ct and ft > 0:
            record["latency_reduction_percent"] = round(100.0 * (ft - ct) / ft, 1)
        else:
            record["latency_reduction_percent"] = None

    return record


def parse_demo_file(path: Path) -> Optional[dict]:
    """
    Read *path*, parse its text, and attach bookkeeping fields.

    Returns None (after printing a warning) when the file cannot be read
    or does not match the expected ContextSearch demo-log format.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        warnings.warn(f"[SKIP] Cannot read {path}: {exc}")
        return None

    record = parse_demo_text(text)
    if record is None:
        warnings.warn(
            f"[SKIP] {path.name} — does not match expected ContextSearch demo format."
        )
        return None

    record["source"]      = path.stem
    record["source_path"] = str(path)
    return record


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame
# ─────────────────────────────────────────────────────────────────────────────

# Canonical column order for the CSV and DataFrame
_COLUMNS = [
    "source",
    "query",
    "corpus_files",
    "loaded_chunks",
    "total_corpus_tokens",
    "full_context_tokens",
    "full_context_chunks",
    "contextsearch_tokens",
    "contextsearch_chunks",
    "tokens_sent_full",
    "tokens_sent_contextsearch",
    "full_time_seconds",
    "contextsearch_time_seconds",
    "token_reduction_percent",
    "latency_reduction_percent",
    "tokens_saved_absolute",
    "chunks_dropped",
]


def build_dataframe(records: list[dict]) -> pd.DataFrame:
    """Convert a list of parsed record dicts into a tidy DataFrame."""
    df = pd.DataFrame(records)
    cols = [c for c in _COLUMNS if c in df.columns]
    return df[cols].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Annotation label helper
# ─────────────────────────────────────────────────────────────────────────────

def _label(row: pd.Series, max_len: int = 32) -> str:
    """
    Return a short string to annotate a chart data point.
    Prefers the first `max_len` characters of the query; falls back to the
    source filename stem when the query is empty.
    """
    q = str(row.get("query", "")).strip()
    if q:
        return (q[:max_len] + "…") if len(q) > max_len else q
    return str(row.get("source", ""))


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    color: str = "steelblue",
    annotate: bool = True,
) -> None:
    """
    Generic scatter plot of *y_col* vs *x_col*.

    Points with missing values in either column are silently skipped.
    Each visible point is annotated with a short query / filename label.
    """
    subset = df[[x_col, y_col]].dropna()
    if subset.empty:
        warnings.warn(f"[SKIP PLOT] No complete data for '{title}'.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        subset[x_col],
        subset[y_col],
        color=color,
        s=80,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )

    if annotate:
        for _, row in df.iterrows():
            xv, yv = row.get(x_col), row.get(y_col)
            if pd.notna(xv) and pd.notna(yv):
                ax.annotate(
                    _label(row),
                    (xv, yv),
                    textcoords="offset points",
                    xytext=(7, 4),
                    fontsize=8,
                    clip_on=True,
                )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_comparison_scatter(df: pd.DataFrame, output_path: Path) -> None:
    """
    Combined scatter plot with both methods on the same axes.

    Full Context points are shown as red circles.
    ContextSearch points are shown as green triangles.
    Paired points (same query) are connected with grey arrows to make the
    reduction in tokens and latency visually obvious.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    FC_COLOR = "tomato"
    CS_COLOR = "mediumseagreen"

    fc_legend_added = False
    cs_legend_added = False

    for _, row in df.iterrows():
        fc_x = row.get("tokens_sent_full")
        fc_y = row.get("full_time_seconds")
        cs_x = row.get("tokens_sent_contextsearch")
        cs_y = row.get("contextsearch_time_seconds")
        lbl  = _label(row)

        if pd.notna(fc_x) and pd.notna(fc_y):
            ax.scatter(
                fc_x, fc_y,
                color=FC_COLOR, s=80, zorder=4,
                label="Full Context" if not fc_legend_added else "",
            )
            ax.annotate(
                lbl, (fc_x, fc_y),
                textcoords="offset points", xytext=(7, 4),
                fontsize=7, color=FC_COLOR, clip_on=True,
            )
            fc_legend_added = True

        if pd.notna(cs_x) and pd.notna(cs_y):
            ax.scatter(
                cs_x, cs_y,
                color=CS_COLOR, s=80, marker="^", zorder=4,
                label="ContextSearch" if not cs_legend_added else "",
            )
            ax.annotate(
                lbl, (cs_x, cs_y),
                textcoords="offset points", xytext=(7, -13),
                fontsize=7, color=CS_COLOR, clip_on=True,
            )
            cs_legend_added = True

        # Arrow from Full Context point → ContextSearch point
        if all(pd.notna(v) for v in [fc_x, fc_y, cs_x, cs_y]):
            ax.annotate(
                "",
                xy=(cs_x, cs_y),
                xytext=(fc_x, fc_y),
                arrowprops=dict(arrowstyle="->", color="grey", lw=0.9),
            )

    ax.legend(fontsize=9)
    ax.set_title(
        "Full Context vs ContextSearch — Tokens Sent & Latency",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Tokens sent to LLM", fontsize=11)
    ax.set_ylabel("Latency (seconds)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_bar_plot(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    xlabel: str,
    output_path: Path,
    color: str = "cornflowerblue",
) -> None:
    """
    Horizontal bar chart of *value_col* indexed by query / file label.

    Horizontal orientation keeps long query-text labels legible.
    The exact value is printed at the end of each bar.
    """
    subset = df[df[value_col].notna()].copy()
    if subset.empty:
        warnings.warn(f"[SKIP PLOT] No data for '{title}'.")
        return

    labels = [_label(row) for _, row in subset.iterrows()]
    values = subset[value_col].tolist()

    fig_height = max(3.0, 0.55 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    y_pos = range(len(labels))

    bars = ax.barh(list(y_pos), values, color=color)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.bar_label(bars, fmt="%.1f", padding=5, fontsize=9)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, index=False)
    print(f"  Saved CSV: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Printed summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    """Print a human-readable aggregate summary to stdout."""

    def avg(col: str) -> str:
        if col in df.columns and df[col].notna().any():
            return f"{df[col].mean():.2f}"
        return "N/A"

    print()
    print("=" * 60)
    print("  ContextSearch Demo Logs — Aggregate Summary")
    print("=" * 60)
    print(f"  Files parsed                  : {len(df)}")
    print(f"  Avg full-context latency      : {avg('full_time_seconds')} s")
    print(f"  Avg ContextSearch latency     : {avg('contextsearch_time_seconds')} s")
    print(f"  Avg token savings             : {avg('token_reduction_percent')} %")
    print(f"  Avg latency reduction         : {avg('latency_reduction_percent')} %")
    print(f"  Avg tokens saved (absolute)   : {avg('tokens_saved_absolute')}")
    print(f"  Avg chunks dropped            : {avg('chunks_dropped')}")
    print("=" * 60)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Input file discovery
# ─────────────────────────────────────────────────────────────────────────────

def collect_paths(input_arg: str) -> list[Path]:
    """
    Resolve the CLI argument to a list of markdown file paths.

    Accepts:
      • A single .md file path  →  returns [that path]
      • A directory path        →  returns sorted list of all *.md files in it
    """
    p = Path(input_arg)
    if p.is_file():
        return [p]
    if p.is_dir():
        paths = sorted(p.glob("*.md"))
        if not paths:
            print(f"Warning: no .md files found in '{p}'.", file=sys.stderr)
        return paths
    print(f"Error: '{input_arg}' is not a valid file or directory.", file=sys.stderr)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse ContextSearch demo markdown logs and produce visualisations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        help="Path to a single .md file or a directory of .md files.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="plots",
        metavar="DIR",
        help="Destination folder for PNG plots and CSV (default: plots/).",
    )
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Skip the combined Full Context / ContextSearch comparison scatter.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Parse all matching files ──────────────────────────────────────────────
    paths   = collect_paths(args.input)
    records = [r for p in paths if (r := parse_demo_file(p)) is not None]

    if not records:
        print("No valid ContextSearch demo records found. Exiting.", file=sys.stderr)
        sys.exit(1)

    df = build_dataframe(records)
    print(f"\nParsed {len(df)} file(s).\n")

    # ── CSV ───────────────────────────────────────────────────────────────────
    save_csv(df, output_dir / "contextsearch_metrics.csv")

    # ── Scatter 1: Full Context — tokens sent vs latency ─────────────────────
    make_scatter_plot(
        df,
        x_col="tokens_sent_full",
        y_col="full_time_seconds",
        title="Full Context — Tokens Sent vs Latency",
        xlabel="Tokens sent to LLM",
        ylabel="Latency (seconds)",
        output_path=output_dir / "scatter_full_context.png",
        color="tomato",
    )

    # ── Scatter 2: ContextSearch — tokens sent vs latency ────────────────────
    make_scatter_plot(
        df,
        x_col="tokens_sent_contextsearch",
        y_col="contextsearch_time_seconds",
        title="ContextSearch — Tokens Sent vs Latency",
        xlabel="Tokens sent to LLM",
        ylabel="Latency (seconds)",
        output_path=output_dir / "scatter_contextsearch.png",
        color="mediumseagreen",
    )

    # ── Scatter 3: Combined comparison (optional) ─────────────────────────────
    if not args.no_comparison:
        make_comparison_scatter(df, output_dir / "scatter_comparison.png")

    # ── Bar 4: Token savings % per query ─────────────────────────────────────
    make_bar_plot(
        df,
        value_col="token_reduction_percent",
        title="Token Savings per Query (%)",
        xlabel="Token reduction (%)",
        output_path=output_dir / "bar_token_savings.png",
        color="cornflowerblue",
    )

    # ── Bar 5: Latency reduction % per query ─────────────────────────────────
    make_bar_plot(
        df,
        value_col="latency_reduction_percent",
        title="Latency Reduction per Query (%)",
        xlabel="Latency reduction (%)",
        output_path=output_dir / "bar_latency_reduction.png",
        color="mediumpurple",
    )

    # ── Printed summary ───────────────────────────────────────────────────────
    print_summary(df)


if __name__ == "__main__":
    main()
