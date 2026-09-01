"""
Expert pilot evaluation: descriptive summaries, omnibus testing, pairwise testing,
and grouped bar plots for the three expert-rated model variants.

Expected inputs are three CSV files corresponding to:
- Base model
- EPS without D2
- EPS

Each CSV should contain seven question columns (Q1..Q7 or equivalent) with A/B/C/D
grades or numeric 0-3 scores. If a shared identifier column is available (for
example ``rater_id`` or ``编号``), the script will align rows for paired testing.

Outputs:
- <prefix>_means_ci.csv
- <prefix>_aligned_scores.csv
- <prefix>_friedman_tests.csv
- <prefix>_wilcoxon_pairwise_tests.csv
- <prefix>_bar_mean_ci.pdf
- <prefix>_bar_mean_ci.png
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DIMENSIONS: List[str] = [
    "consensus",
    "correctness",
    "completeness",
    "unbiasedness",
    "clarity",
    "empathy",
    "actionability",
]
QNUM_TO_DIM: Dict[int, str] = {i + 1: dim for i, dim in enumerate(DIMENSIONS)}
LETTER_TO_SCORE: Dict[str, int] = {"A": 0, "B": 1, "C": 2, "D": 3}

PAIRWISE_COMPARISONS: List[Tuple[str, str]] = [
    ("EPS", "EPS without D2"),
    ("EPS without D2", "Base model"),
    ("EPS", "Base model"),
]

ID_CANDIDATES: List[str] = ["编号", "rater_id", "RaterID", "ID", "id"]


def configure_matplotlib_fonts() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 10,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv_robust(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def parse_score(x) -> float:
    if pd.isna(x):
        return np.nan

    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x)
        return v if 0.0 <= v <= 3.0 else np.nan

    s = str(x).strip()

    m = re.match(r"^([ABCD])\s*[\.\．:：\)\）\]、】【、。]?", s, flags=re.IGNORECASE)
    if m:
        return float(LETTER_TO_SCORE[m.group(1).upper()])

    m = re.match(r"^([ABCD])$", s, flags=re.IGNORECASE)
    if m:
        return float(LETTER_TO_SCORE[m.group(1).upper()])

    m = re.match(r"^\s*([0-3])(?:\.0+)?\s*$", s)
    if m:
        return float(m.group(1))

    return np.nan


def mean_ci_t(x, alpha: float = 0.05) -> Tuple[float, float, float, int, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2:
        return np.nan, np.nan, np.nan, int(n), np.nan

    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / np.sqrt(n)

    try:
        from scipy.stats import t

        tcrit = float(t.ppf(1 - alpha / 2, df=n - 1))
    except Exception:
        tcrit = 1.959963984540054

    lo = mean - tcrit * se
    hi = mean + tcrit * se
    return mean, lo, hi, int(n), sd


def pick_q1_to_q7_cols(df: pd.DataFrame) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for col in df.columns:
        s = str(col).strip()

        m = re.match(r"^([1-7])\s*[\.\．]", s)
        if m:
            mapping.setdefault(int(m.group(1)), col)
            continue

        m = re.match(r"^[Qq]\s*([1-7])\s*$", s)
        if m:
            mapping.setdefault(int(m.group(1)), col)
            continue

        m = re.match(r"^([1-7])\s*$", s)
        if m:
            mapping.setdefault(int(m.group(1)), col)

    missing = [q for q in range(1, 8) if q not in mapping]
    if missing:
        raise ValueError(
            f"Cannot find question columns for Q{missing}. "
            "Expected headers like '1.'...'7.' or 'Q1'...'Q7'."
        )
    return mapping


def holm_adjust(pvals) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)

    mask = ~np.isnan(pvals)
    if not np.any(mask):
        return out

    pv = pvals[mask]
    m = int(pv.size)
    order = np.argsort(pv)

    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for k, idx in enumerate(order):
        val = (m - k) * pv[idx]
        running_max = max(running_max, val)
        adj[idx] = min(running_max, 1.0)

    out[mask] = adj
    return out


def detect_pair_key(dfs: List[pd.DataFrame], user_key: Optional[str]) -> Optional[str]:
    if user_key is not None:
        if all(user_key in df.columns for df in dfs):
            return user_key
        return None

    common_cols = set(dfs[0].columns)
    for df in dfs[1:]:
        common_cols &= set(df.columns)

    for key in ID_CANDIDATES:
        if key in common_cols:
            return key
    return None


def build_numeric_score_frame(df: pd.DataFrame, pair_key: Optional[str]) -> pd.DataFrame:
    qcols = pick_q1_to_q7_cols(df)

    out = pd.DataFrame()
    if pair_key is not None:
        out[pair_key] = df[pair_key]

    for q in range(1, 8):
        out[f"Q{q}"] = df[qcols[q]].map(parse_score).astype(float)

    if pair_key is not None:
        out = out.dropna(subset=[pair_key]).copy()
        out = out.drop_duplicates(subset=[pair_key], keep="first").copy()

    return out


def align_score_frames(score_frames: Dict[str, pd.DataFrame], pair_key: Optional[str]) -> pd.DataFrame:
    labels = list(score_frames.keys())

    if pair_key is not None:
        renamed = []
        for label in labels:
            df = score_frames[label].copy()
            rename_map = {f"Q{q}": f"{label}__Q{q}" for q in range(1, 8)}
            df = df.rename(columns=rename_map)
            renamed.append(df)

        merged = renamed[0]
        for df in renamed[1:]:
            merged = merged.merge(df, on=pair_key, how="inner")
        return merged

    lengths = [len(score_frames[label]) for label in labels]
    if len(set(lengths)) != 1:
        raise ValueError(
            "No common pairing column found, and row counts differ across files. "
            "Please add a common ID column such as '编号' or specify --pair-key."
        )

    merged = pd.DataFrame({"_row_id": np.arange(lengths[0])})
    for label in labels:
        df = score_frames[label].reset_index(drop=True).copy()
        rename_map = {f"Q{q}": f"{label}__Q{q}" for q in range(1, 8)}
        df = df.rename(columns=rename_map)
        keep_cols = [f"{label}__Q{q}" for q in range(1, 8)]
        merged = pd.concat([merged, df[keep_cols]], axis=1)
    return merged


def summarize_model(df: pd.DataFrame, label: str) -> pd.DataFrame:
    qcols = pick_q1_to_q7_cols(df)
    rows = []
    for q in range(1, 8):
        dim = QNUM_TO_DIM[q]
        col = qcols[q]
        scores = df[col].map(parse_score).astype(float)
        mean, lo, hi, n, sd = mean_ci_t(scores.values, alpha=0.05)
        rows.append(
            {
                "model": label,
                "dimension": dim,
                "question": q,
                "n": n,
                "mean": mean,
                "sd": sd,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    return pd.DataFrame(rows).sort_values("question")


def safe_friedman(x1, x2, x3) -> Tuple[float, float, int]:
    d = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3}).dropna()
    n = int(len(d))
    if n < 2:
        return np.nan, np.nan, n

    if np.allclose(d["x1"], d["x2"]) and np.allclose(d["x2"], d["x3"]):
        return 0.0, 1.0, n

    try:
        from scipy.stats import friedmanchisquare

        stat, p = friedmanchisquare(d["x1"], d["x2"], d["x3"])
        return float(stat), float(p), n
    except Exception:
        return np.nan, np.nan, n


def safe_wilcoxon(x, y) -> Tuple[float, float, int, float]:
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    n = int(len(d))
    if n < 1:
        return np.nan, np.nan, n, np.nan

    diff = d["x"] - d["y"]
    mean_diff = float(diff.mean())

    if np.allclose(diff, 0.0):
        return 0.0, 1.0, n, mean_diff

    try:
        from scipy.stats import wilcoxon

        stat, p = wilcoxon(
            d["x"],
            d["y"],
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )
        return float(stat), float(p), n, mean_diff
    except Exception:
        return np.nan, np.nan, n, mean_diff


def build_friedman_results(aligned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q in range(1, 8):
        stat, p, n = safe_friedman(
            aligned[f"Base model__Q{q}"],
            aligned[f"EPS without D2__Q{q}"],
            aligned[f"EPS__Q{q}"],
        )
        rows.append(
            {
                "dimension": QNUM_TO_DIM[q],
                "question": q,
                "n_used": n,
                "friedman_chi2": stat,
                "p_value": p,
            }
        )

    out = pd.DataFrame(rows).sort_values("question").reset_index(drop=True)
    out["p_holm_7dims"] = holm_adjust(out["p_value"].to_numpy())
    return out


def build_pairwise_results(aligned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q in range(1, 8):
        for model_a, model_b in PAIRWISE_COMPARISONS:
            stat, p, n, mean_diff = safe_wilcoxon(
                aligned[f"{model_a}__Q{q}"],
                aligned[f"{model_b}__Q{q}"],
            )
            rows.append(
                {
                    "dimension": QNUM_TO_DIM[q],
                    "question": q,
                    "comparison": f"{model_a} - {model_b}",
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_used": n,
                    "mean_diff": mean_diff,
                    "wilcoxon_stat": stat,
                    "p_value": p,
                }
            )

    out = pd.DataFrame(rows).sort_values(["question", "comparison"]).reset_index(drop=True)

    out["p_holm_7dims_within_comparison"] = np.nan
    for comp in out["comparison"].unique():
        mask = out["comparison"] == comp
        out.loc[mask, "p_holm_7dims_within_comparison"] = holm_adjust(
            out.loc[mask, "p_value"].to_numpy()
        )

    out["p_holm_3pairs_within_dimension"] = np.nan
    for q in out["question"].unique():
        mask = out["question"] == q
        out.loc[mask, "p_holm_3pairs_within_dimension"] = holm_adjust(
            out.loc[mask, "p_value"].to_numpy()
        )

    return out


def plot_grouped_bars(summary: pd.DataFrame, out_pdf: Path, out_png: Path, show: bool) -> None:
    dim_order = DIMENSIONS
    bar_order = ["Base model", "EPS without D2", "EPS"]

    pivot_mean = summary.pivot(index="dimension", columns="model", values="mean").loc[dim_order, bar_order]
    pivot_lo = summary.pivot(index="dimension", columns="model", values="ci_low").loc[dim_order, bar_order]
    pivot_hi = summary.pivot(index="dimension", columns="model", values="ci_high").loc[dim_order, bar_order]

    means = pivot_mean.values
    yerr_lower = means - pivot_lo.values
    yerr_upper = pivot_hi.values - means
    yerr = np.stack([yerr_lower, yerr_upper], axis=0)

    facecolors = ["#E59693", "#FCD7AF", "#989CC8"]
    errcolors = ["#C93735", "#F8AB61", "#333A8C"]

    n_dims = len(dim_order)
    n_models = len(bar_order)
    x = np.arange(n_dims)
    width = 0.24

    fig, ax = plt.subplots(figsize=(10.5, 4.2), dpi=200)

    for j, model in enumerate(bar_order):
        ax.bar(
            x + (j - (n_models - 1) / 2) * width,
            pivot_mean[model].values,
            width=width,
            label=model,
            color=facecolors[j],
            edgecolor="none",
            linewidth=0,
            yerr=yerr[:, :, j],
            capsize=3,
            error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": errcolors[j]},
        )

    ax.set_ylabel("Evaluation Scores")
    ax.set_ylim(0, 3.05)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in dim_order], rotation=25, ha="right")

    ax.yaxis.grid(True, linewidth=0.6, alpha=0.30)
    ax.xaxis.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    legend_order = [labels.index("Base model"), labels.index("EPS without D2"), labels.index("EPS")]
    ax.legend(
        [handles[i] for i in legend_order],
        [labels[i] for i in legend_order],
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Expert pilot evaluation (Fig. 5b): descriptive means with 95% CI, Friedman omnibus "
            "tests, paired Wilcoxon pairwise tests, Holm adjustment, and grouped bar plots."
        )
    )
    p.add_argument("--base", type=str, default=None, help="Path to base model CSV.")
    p.add_argument("--d1", type=str, default=None, help="Path to EPS-without-D2 CSV.")
    p.add_argument("--eps", type=str, default=None, help="Path to EPS CSV.")
    p.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing base_model.csv, eps_without_d2.csv, eps.csv.",
    )
    p.add_argument(
        "--pair-key",
        type=str,
        default=None,
        help="Column name used to align paired raters across files. Auto-detected when omitted.",
    )
    p.add_argument("--no-tests", action="store_true", help="Skip Friedman and Wilcoxon tests.")
    p.add_argument("--no-show", action="store_true", help="Do not display the plot window.")
    p.add_argument("--outdir", type=str, default=".", help="Output directory.")
    p.add_argument("--prefix", type=str, default="expert_pilot", help="Output filename prefix.")
    return p


def resolve_inputs(args: argparse.Namespace) -> Dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    input_dir = Path(args.input_dir) if args.input_dir else (repo_root / "data" / "expert_pilot")

    base = Path(args.base) if args.base else (input_dir / "base_model.csv")
    d1 = Path(args.d1) if args.d1 else (input_dir / "eps_without_d2.csv")
    eps = Path(args.eps) if args.eps else (input_dir / "eps.csv")

    missing = [str(path) for path in (base, d1, eps) if not path.exists()]
    if missing:
        msg = (
            "Missing input file(s):\n- "
            + "\n- ".join(missing)
            + "\n\nProvide explicit paths via --base/--d1/--eps, or place files under data/expert_pilot/:\n"
            + "  base_model.csv\n  eps_without_d2.csv\n  eps.csv\n"
        )
        raise FileNotFoundError(msg)

    return {"Base model": base, "EPS without D2": d1, "EPS": eps}


def build_alignment_and_tests(
    raw_frames: Dict[str, pd.DataFrame], pair_key: Optional[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_frames = {
        label: build_numeric_score_frame(df, pair_key=pair_key) for label, df in raw_frames.items()
    }
    aligned = align_score_frames(score_frames, pair_key=pair_key)
    friedman_results = build_friedman_results(aligned)
    pairwise_results = build_pairwise_results(aligned)
    return aligned, friedman_results, pairwise_results


def main() -> int:
    args = build_parser().parse_args()
    configure_matplotlib_fonts()

    try:
        paths = resolve_inputs(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    raw_frames: Dict[str, pd.DataFrame] = {}
    for label, path in paths.items():
        df = read_csv_robust(path)
        df.columns = [str(col).strip() for col in df.columns]
        raw_frames[label] = df

    pair_key = detect_pair_key(list(raw_frames.values()), user_key=args.pair_key)

    summaries = []
    for label, df in raw_frames.items():
        frame = df.copy()
        if pair_key and pair_key in frame.columns:
            frame = frame.dropna(subset=[pair_key]).copy()
        summaries.append(summarize_model(frame, label))
    summary = pd.concat(summaries, ignore_index=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_summary = outdir / f"{args.prefix}_means_ci.csv"
    summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
    print(f"Saved summary: {out_summary}")

    aligned = None
    friedman_results = None
    pairwise_results = None
    if not args.no_tests:
        try:
            aligned, friedman_results, pairwise_results = build_alignment_and_tests(
                raw_frames=raw_frames, pair_key=pair_key
            )
        except Exception as exc:
            print(f"Error while building paired tests: {exc}", file=sys.stderr)
            return 3

        out_aligned = outdir / f"{args.prefix}_aligned_scores.csv"
        out_friedman = outdir / f"{args.prefix}_friedman_tests.csv"
        out_pairwise = outdir / f"{args.prefix}_wilcoxon_pairwise_tests.csv"

        aligned.to_csv(out_aligned, index=False, encoding="utf-8-sig")
        friedman_results.to_csv(out_friedman, index=False, encoding="utf-8-sig")
        pairwise_results.to_csv(out_pairwise, index=False, encoding="utf-8-sig")

        print(f"Saved aligned scores: {out_aligned}")
        print(f"Saved Friedman tests: {out_friedman}")
        print(f"Saved Wilcoxon tests: {out_pairwise}")

        if pair_key is not None:
            print(f"Paired alignment used common ID column: {pair_key}")
        else:
            print("No common ID column detected; paired alignment used row order.")

    out_pdf = outdir / f"{args.prefix}_bar_mean_ci.pdf"
    out_png = outdir / f"{args.prefix}_bar_mean_ci.png"
    plot_grouped_bars(summary=summary, out_pdf=out_pdf, out_png=out_png, show=(not args.no_show))
    print(f"Saved: {out_pdf} and {out_png}")

    if friedman_results is not None and pairwise_results is not None:
        print("\n=== Friedman omnibus tests ===")
        print(friedman_results.to_string(index=False))
        print("\n=== Paired Wilcoxon pairwise tests ===")
        print(pairwise_results.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
