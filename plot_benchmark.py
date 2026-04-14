"""
plot_benchmark.py
=================
Read a ContextSearch benchmark JSON file and produce two scatter plots:
  1. Prompt Tokens vs Markdown Files Searched
  2. Latency vs Markdown Files Searched

Usage:
    python plot_benchmark.py path/to/results.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_runs(json_path: Path) -> list[dict]:
    """Load and validate the runs array from a benchmark JSON file."""
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if "runs" not in data:
        raise ValueError(f"No 'runs' key found in {json_path}")
    return data["runs"]


def split_runs_by_mode(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (vanilla_runs, enhanced_runs) split by the 'mode' field."""
    vanilla = [r for r in runs if r.get("mode") == "vanilla"]
    enhanced = [r for r in runs if r.get("mode") == "enhanced"]
    return vanilla, enhanced


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _extract(runs: list[dict], x_field: str, y_field: str) -> tuple[list, list]:
    """Extract paired (x, y) values, skipping rows where either field is None."""
    xs, ys = [], []
    for r in runs:
        x = r.get(x_field)
        y = r.get(y_field)
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    return xs, ys


def plot_tokens_vs_md_files(
    vanilla: list[dict],
    enhanced: list[dict],
    output_path: Path,
) -> None:
    """Scatter plot: md_files_searched vs prompt_tokens_sent."""
    vx, vy = _extract(vanilla, "md_files_searched", "prompt_tokens_sent")
    ex, ey = _extract(enhanced, "md_files_searched", "prompt_tokens_sent")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(vx, vy, color="red",  marker="o", label="vanilla",  zorder=3)
    ax.scatter(ex, ey, color="blue", marker="o", label="enhanced", zorder=3)

    ax.set_title("Prompt Tokens vs Markdown Files Searched")
    ax.set_xlabel("Markdown Files Searched")
    ax.set_ylabel("Prompt Tokens Sent")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")


def plot_latency_vs_md_files(
    vanilla: list[dict],
    enhanced: list[dict],
    output_path: Path,
) -> None:
    """Scatter plot: md_files_searched vs llm_latency_seconds."""
    vx, vy = _extract(vanilla, "md_files_searched", "llm_latency_seconds")
    ex, ey = _extract(enhanced, "md_files_searched", "llm_latency_seconds")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(vx, vy, color="red",  marker="o", label="vanilla",  zorder=3)
    ax.scatter(ex, ey, color="blue", marker="o", label="enhanced", zorder=3)

    ax.set_title("Latency vs Markdown Files Searched")
    ax.set_xlabel("Markdown Files Searched")
    ax.set_ylabel("LLM Latency (seconds)")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ContextSearch benchmark JSON results")
    parser.add_argument("json_file", type=Path, help="Path to the benchmark JSON file")
    args = parser.parse_args()

    json_path: Path = args.json_file.resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    runs = load_runs(json_path)
    vanilla, enhanced = split_runs_by_mode(runs)

    out_dir = json_path.parent
    stem = json_path.stem

    plot_tokens_vs_md_files(vanilla, enhanced, out_dir / f"{stem}_tokens.png")
    plot_latency_vs_md_files(vanilla, enhanced, out_dir / f"{stem}_latency.png")

    plt.show(block=False)
    input("Press Enter to close plots…")


if __name__ == "__main__":
    main()
