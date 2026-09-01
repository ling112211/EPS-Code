"""
Benchmark accuracy bar chart (Fig. 5a).

Reads a pre-computed CSV produced by evaluate_benchmark.py and draws a
four-panel grouped bar chart (one panel per benchmark dataset) showing
mean accuracy with 95% CI error bars for every model.

Expected CSV columns:
    group               – model group label, e.g. "Base model", "EPS", "Mainstream LLMs"
    model               – model name, e.g. "Qwen3-14B"
    benchmark           – benchmark name: CMB | CMExam | MedMCQA | MedQA
    mean_accuracy_pct   – mean accuracy in percentage points (0–100)
    ci_low_pct          – lower bound of 95% CI (percentage points)
    ci_high_pct         – upper bound of 95% CI (percentage points)
    n_questions         – number of questions evaluated (informational)

Usage
-----
    python benchmark/plot_benchmark.py \
        --input  data/benchmark_results/benchmark_accuracy.csv \
        --outdir outputs/benchmark
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARKS: List[str] = ["CMB", "CMExam", "MedMCQA", "MedQA"]

# Display order of groups (left to right within each panel)
GROUP_ORDER: List[str] = ["Base model", "EPS", "Mainstream LLMs"]

# Model display order within each group
MODEL_ORDER: Dict[str, List[str]] = {
    "Base model": [
        "DeepSeek-R1-8B",
        "Qwen3-8B",
        "DeepSeek-R1-14B",
        "Qwen3-14B",
    ],
    "EPS": [
        "DeepSeek-R1-8B",
        "Qwen3-8B",
        "DeepSeek-R1-14B",
        "Qwen3-14B",
    ],
    "Mainstream LLMs": [
        "ChatGPT-5",
        "DeepSeek-R1",
        "Gemini 2.5 Flash",
        "Grok 4 Fast",
    ],
}

# Short tick labels (used on x-axis)
SHORT_NAMES: Dict[str, str] = {
    "DeepSeek-R1-8B":   "DS-R1\n8B",
    "Qwen3-8B":         "Qwen3\n8B",
    "DeepSeek-R1-14B":  "DS-R1\n14B",
    "Qwen3-14B":        "Qwen3\n14B",
    "ChatGPT-5":        "GPT-5",
    "DeepSeek-R1":      "DS-R1",
    "Gemini 2.5 Flash": "Gemini\n2.5F",
    "Grok 4 Fast":      "Grok\n4F",
}

# Colours – Base model light blue, EPS dark blue, Mainstream green
GROUP_COLORS: Dict[str, str] = {
    "Base model":      "#9ecae1",
    "EPS":             "#2171b5",
    "Mainstream LLMs": "#74c476",
}

BAR_WIDTH: float = 0.20
INNER_GAP: float = 0.04   # gap between bars within a group
GROUP_GAP: float = 0.45   # gap between groups


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    required = {"group", "model", "benchmark", "mean_accuracy_pct", "ci_low_pct", "ci_high_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df


def _lookup(
    df: pd.DataFrame,
    group: str,
    model: str,
    benchmark: str,
) -> Optional[tuple[float, float, float]]:
    """Return (mean, ci_low, ci_high) in percentage points, or None if not found."""
    mask = (df["group"] == group) & (df["model"] == model) & (df["benchmark"] == benchmark)
    rows = df[mask]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return float(row["mean_accuracy_pct"]), float(row["ci_low_pct"]), float(row["ci_high_pct"])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_benchmark(df: pd.DataFrame, outdir: Path) -> None:
    mpl.rcParams.update({
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "axes.axisbelow":    True,
        "grid.linestyle":    "--",
        "grid.alpha":        0.45,
    })

    fig, axes = plt.subplots(1, 4, figsize=(20, 7))
    fig.suptitle(
        "Benchmark Accuracy across Medical QA Datasets",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    for ax, bmark in zip(axes, BENCHMARKS):
        x_pos: float = 0.0
        xtick_positions: List[float] = []
        xtick_labels: List[str] = []
        group_centers: List[float] = []

        for grp in GROUP_ORDER:
            models = MODEL_ORDER[grp]
            bar_positions: List[float] = []

            for model in models:
                result = _lookup(df, grp, model, bmark)

                if result is None:
                    # Reserve space even if data is absent
                    x_pos += BAR_WIDTH + INNER_GAP
                    bar_positions.append(x_pos)
                    continue

                mean, ci_lo, ci_hi = result
                yerr_lo = max(mean - ci_lo, 0.0)
                yerr_hi = max(ci_hi - mean, 0.0)

                ax.bar(
                    x_pos, mean,
                    width=BAR_WIDTH,
                    color=GROUP_COLORS[grp],
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=3,
                    yerr=[[yerr_lo], [yerr_hi]],
                    error_kw=dict(elinewidth=1.0, capsize=2.5, ecolor="#444444", zorder=4),
                )

                bar_positions.append(x_pos)
                xtick_positions.append(x_pos)
                xtick_labels.append(SHORT_NAMES.get(model, model))

                x_pos += BAR_WIDTH + INNER_GAP

            if bar_positions:
                group_centers.append((bar_positions[0] + bar_positions[-1]) / 2.0)

            x_pos += GROUP_GAP - INNER_GAP  # separate groups

        # Axes formatting
        ax.set_title(bmark, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylim(0, 108)
        ax.set_ylabel("Accuracy (%)" if ax is axes[0] else "", fontsize=9)
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, fontsize=6.5, rotation=40, ha="right")

        # Group labels below x-axis
        y_label = ax.get_ylim()[0] - 13
        for cx, grp in zip(group_centers, GROUP_ORDER):
            ax.text(
                cx, y_label, grp,
                ha="center", va="top",
                fontsize=7, fontweight="bold",
                color=GROUP_COLORS[grp],
                clip_on=False,
            )

    # Legend
    legend_handles = [
        Patch(facecolor=GROUP_COLORS[g], label=g) for g in GROUP_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.96),
        fontsize=9,
        frameon=False,
    )

    plt.subplots_adjust(bottom=0.26, top=0.90, wspace=0.30)

    outdir.mkdir(parents=True, exist_ok=True)
    png_path = outdir / "benchmark_accuracy.png"
    pdf_path = outdir / "benchmark_accuracy.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot benchmark accuracy bar chart (Fig. 5a) from a pre-computed CSV."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/benchmark_results/benchmark_accuracy.csv"),
        help="Path to the benchmark accuracy CSV file (default: data/benchmark_results/benchmark_accuracy.csv).",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/benchmark"),
        help="Directory where PNG and PDF figures are saved (default: outputs/benchmark).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_csv(args.input)
    plot_benchmark(df, args.outdir)


if __name__ == "__main__":
    main()
