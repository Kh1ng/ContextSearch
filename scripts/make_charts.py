"""
Generate presentation-quality charts for slides 6 & 7.
Reads demo/output/*.json and results/eval_results.json.
Outputs PNG files to docs/charts/.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "docs" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette ─────────────────────────────────────────────────────────────────
VANILLA_COLOR  = "#D64045"   # red  – full context / expensive
ENHANCED_COLOR = "#2E86AB"   # blue – ContextSearch / optimised
BG             = "#FAFAFA"
GRID_COLOR     = "#E0E0E0"
FONT_TITLE     = 14
FONT_LABEL     = 11
FONT_TICK      = 9

def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.spines[["top","right"]].set_visible(False)

# ── 1. Real-World: Token comparison as corpus grows ──────────────────────────
def chart_rw_tokens(runs: list[dict], out: Path):
    vanilla  = [(r["md_files_searched"], r["prompt_tokens_sent"])
                for r in runs if r["mode"] == "vanilla"]
    enhanced = [(r["md_files_searched"], r["prompt_tokens_sent"])
                for r in runs if r["mode"] == "enhanced"]

    vanilla.sort();  enhanced.sort()
    x   = [v[0] for v in vanilla]
    yv  = [v[1] for v in vanilla]
    ye  = [e[1] for e in enhanced]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, yv, "o-", color=VANILLA_COLOR,  linewidth=2.5, markersize=7,
            label="Full Context (no filtering)")
    ax.plot(x, ye, "s-", color=ENHANCED_COLOR, linewidth=2.5, markersize=7,
            label="ContextSearch (A* filtered)")
    ax.fill_between(x, ye, yv, alpha=0.12, color=ENHANCED_COLOR)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xticks(x)
    _style(ax, "Token Usage as Agent Corpus Grows", "Number of Skill Files", "Tokens Sent to LLM")
    ax.legend(fontsize=FONT_TICK, framealpha=0.9)

    # annotate last pair
    ax.annotate(f"{int(yv[-1]):,}", (x[-1], yv[-1]), textcoords="offset points",
                xytext=(6, 4), fontsize=8, color=VANILLA_COLOR, fontweight="bold")
    ax.annotate(f"{int(ye[-1]):,}", (x[-1], ye[-1]), textcoords="offset points",
                xytext=(6, -14), fontsize=8, color=ENHANCED_COLOR, fontweight="bold")

    fig.patch.set_facecolor(BG)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ── 2. Real-World: Latency comparison as corpus grows ────────────────────────
def chart_rw_latency(runs: list[dict], out: Path):
    vanilla  = [(r["md_files_searched"], r["llm_latency_seconds"])
                for r in runs if r["mode"] == "vanilla"]
    enhanced = [(r["md_files_searched"], r["llm_latency_seconds"])
                for r in runs if r["mode"] == "enhanced"]

    vanilla.sort();  enhanced.sort()
    x   = [v[0] for v in vanilla]
    yv  = [v[1] for v in vanilla]
    ye  = [e[1] for e in enhanced]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, yv, "o-", color=VANILLA_COLOR,  linewidth=2.5, markersize=7,
            label="Full Context")
    ax.plot(x, ye, "s-", color=ENHANCED_COLOR, linewidth=2.5, markersize=7,
            label="ContextSearch")
    ax.fill_between(x, ye, yv, alpha=0.12, color=ENHANCED_COLOR)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}s"))
    ax.set_xticks(x)
    _style(ax, "LLM Response Latency as Agent Corpus Grows", "Number of Skill Files", "Latency (seconds)")
    ax.legend(fontsize=FONT_TICK, framealpha=0.9)

    ax.annotate(f"{yv[-1]:.1f}s", (x[-1], yv[-1]), textcoords="offset points",
                xytext=(6, 4), fontsize=8, color=VANILLA_COLOR, fontweight="bold")
    ax.annotate(f"{ye[-1]:.1f}s", (x[-1], ye[-1]), textcoords="offset points",
                xytext=(6, -14), fontsize=8, color=ENHANCED_COLOR, fontweight="bold")

    fig.patch.set_facecolor(BG)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ── 3. SciFact: Recall@10 + Token Savings across budgets ────────────────────
def chart_scifact(results: list[dict], out: Path):
    results = sorted(results, key=lambda r: r["budget"])
    budgets  = [r["budget"] for r in results]
    recalls  = [r["recall_at_10"] * 100 for r in results]
    savings  = [r["token_savings"] * 100 for r in results]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()

    ax1.bar(budgets, savings, width=[b*0.3 for b in budgets],
            color=ENHANCED_COLOR, alpha=0.7, label="Token Savings %", zorder=3)
    ax2.plot(budgets, recalls, "o-", color=VANILLA_COLOR, linewidth=2.5,
             markersize=7, label="Recall@10 %", zorder=4)

    ax1.set_xscale("log")
    ax1.set_xticks(budgets)
    ax1.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax1.set_xlabel("Token Budget", fontsize=FONT_LABEL)
    ax1.set_ylabel("Token Savings (%)", fontsize=FONT_LABEL, color=ENHANCED_COLOR)
    ax2.set_ylabel("Recall@10 (%)", fontsize=FONT_LABEL, color=VANILLA_COLOR)
    ax1.tick_params(axis="y", labelcolor=ENHANCED_COLOR, labelsize=FONT_TICK)
    ax2.tick_params(axis="y", labelcolor=VANILLA_COLOR,  labelsize=FONT_TICK)
    ax1.tick_params(axis="x", labelsize=FONT_TICK)
    ax1.set_facecolor(BG)
    ax1.set_title("SciFact Benchmark — Token Savings vs Recall@10",
                  fontsize=FONT_TITLE, fontweight="bold", pad=10)
    ax1.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax1.spines[["top"]].set_visible(False)

    # combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=FONT_TICK, framealpha=0.9, loc="center right")

    fig.patch.set_facecolor(BG)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ── 4. Real-world token savings % bar ────────────────────────────────────────
def chart_rw_savings(runs: list[dict], out: Path):
    pairs = {}
    for r in runs:
        k = r["md_files_searched"]
        pairs.setdefault(k, {})[r["mode"]] = r

    xs, savings_pct, latency_pct = [], [], []
    for k in sorted(pairs):
        p = pairs[k]
        if "vanilla" in p and "enhanced" in p:
            v, e = p["vanilla"], p["enhanced"]
            tok_save = (v["prompt_tokens_sent"] - e["prompt_tokens_sent"]) / v["prompt_tokens_sent"] * 100
            lat_save = (v["llm_latency_seconds"] - e["llm_latency_seconds"]) / v["llm_latency_seconds"] * 100
            xs.append(k)
            savings_pct.append(tok_save)
            latency_pct.append(lat_save)

    x = np.arange(len(xs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - w/2, savings_pct, w, label="Token Reduction %",  color=ENHANCED_COLOR, zorder=3)
    b2 = ax.bar(x + w/2, latency_pct, w, label="Latency Reduction %", color="#F4A261", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} file{'s' if n>1 else ''}" for n in xs], fontsize=FONT_TICK)
    ax.bar_label(b1, fmt="%.0f%%", padding=3, fontsize=7, fontweight="bold")
    ax.bar_label(b2, fmt="%.0f%%", padding=3, fontsize=7, fontweight="bold")
    ax.set_ylim(0, 110)
    _style(ax, "Real-World: % Reduction vs Full Context", "Corpus Size", "Reduction (%)")
    ax.legend(fontsize=FONT_TICK, framealpha=0.9)

    fig.patch.set_facecolor(BG)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load best real-world run (131258 has the clean corpus-growth data)
    rw_path = PROJECT_ROOT / "demo" / "output" / "2026.04.12.131258.json"
    with rw_path.open() as f:
        rw = json.load(f)
    runs = rw["runs"]

    # Load SciFact results
    sf_path = PROJECT_ROOT / "results" / "eval_results.json"
    with sf_path.open() as f:
        sf_results = json.load(f)
    if isinstance(sf_results, dict):
        sf_results = [sf_results]

    print("Generating charts…")
    chart_rw_tokens (runs, OUT_DIR / "rw_tokens.png")
    chart_rw_latency(runs, OUT_DIR / "rw_latency.png")
    chart_rw_savings(runs, OUT_DIR / "rw_savings.png")
    chart_scifact   (sf_results, OUT_DIR / "scifact.png")
    print("Done.")
