"""EPS accuracy gains relative to corresponding base models (Fig. 4c)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


BENCHMARKS = ["CMB", "CMExam", "MedMCQA", "MedQA"]
BASE_MODELS = ["DeepSeek-R1-8B", "Qwen3-8B", "DeepSeek-R1-14B", "Qwen3-14B"]
ROW_LABELS = ["DeepSeek-R1\n8B", "Qwen3\n8B", "DeepSeek-R1\n14B", "Qwen3\n14B"]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_gain_matrix(input_csv: Path) -> tuple[np.ndarray, pd.DataFrame]:
    data = pd.read_csv(input_csv)
    required = {"group", "model", "benchmark", "mean_accuracy_pct"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    base = (
        data.loc[data["group"] == "Base model"]
        .pivot(index="model", columns="benchmark", values="mean_accuracy_pct")
        .reindex(index=BASE_MODELS, columns=BENCHMARKS)
    )
    eps = (
        data.loc[data["group"] == "EPS"]
        .pivot(index="model", columns="benchmark", values="mean_accuracy_pct")
        .reindex(index=BASE_MODELS, columns=BENCHMARKS)
    )
    if base.isna().any().any() or eps.isna().any().any():
        raise ValueError("Input CSV does not contain all base/EPS model-benchmark combinations.")

    gain = eps.to_numpy(dtype=float) - base.to_numpy(dtype=float)
    summary_rows = []
    for idx, model in enumerate(BASE_MODELS):
        for col_idx, benchmark in enumerate(BENCHMARKS):
            summary_rows.append(
                {
                    "base_model": model,
                    "eps_model": f"{model} (EPS)",
                    "benchmark": benchmark,
                    "base_accuracy": base.iloc[idx, col_idx],
                    "eps_accuracy": eps.iloc[idx, col_idx],
                    "gain_percentage_points": gain[idx, col_idx],
                }
            )
        summary_rows.append(
            {
                "base_model": model,
                "eps_model": f"{model} (EPS)",
                "benchmark": "Mean",
                "base_accuracy": float(base.iloc[idx].mean()),
                "eps_accuracy": float(eps.iloc[idx].mean()),
                "gain_percentage_points": float(gain[idx].mean()),
            }
        )
    return np.column_stack([gain, gain.mean(axis=1)]), pd.DataFrame(summary_rows)


def plot_gain_heatmap(gains: np.ndarray, out_png: Path, out_pdf: Path) -> None:
    columns = BENCHMARKS + ["Mean"]
    cmap = LinearSegmentedColormap.from_list(
        "eps_gain_blue",
        ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"],
    )
    fig, ax = plt.subplots(figsize=(6.6, 3.2), dpi=300)
    image = ax.imshow(gains, cmap=cmap, vmin=0, vmax=40, aspect="auto")
    ax.set_xticks(np.arange(len(columns)), columns, fontsize=10)
    ax.set_yticks(np.arange(len(BASE_MODELS)), ROW_LABELS, fontsize=10)
    ax.tick_params(axis="both", length=0)
    ax.set_xlabel("Benchmark", fontsize=10, labelpad=8)
    ax.set_ylabel("Base model", fontsize=10, labelpad=8)

    for row in range(gains.shape[0]):
        for col in range(gains.shape[1]):
            value = gains[row, col]
            ax.text(
                col,
                row,
                f"+{value:.1f}",
                ha="center",
                va="center",
                color="white" if value >= 22 else "#1A1A1A",
                fontsize=9.2,
                fontweight="bold" if col == len(columns) - 1 else "normal",
            )

    ax.axvline(len(columns) - 1.5, color="white", linewidth=2.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(BASE_MODELS), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    colorbar.ax.tick_params(labelsize=8, length=2)
    colorbar.set_label("Gain (percentage points)", fontsize=9, labelpad=8)
    colorbar.outline.set_linewidth(0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot EPS accuracy-gain heatmap (Fig. 4c).")
    parser.add_argument("--input", type=Path, default=Path("data/benchmark_results/benchmark_accuracy.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs/benchmark"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    gains, summary = load_gain_matrix(args.input)
    out_png = args.outdir / "benchmark_eps_accuracy_gain_heatmap.png"
    out_pdf = args.outdir / "benchmark_eps_accuracy_gain_heatmap.pdf"
    out_csv = args.outdir / "benchmark_eps_accuracy_gain_summary.csv"
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig", float_format="%.2f")
    plot_gain_heatmap(gains, out_png, out_pdf)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
