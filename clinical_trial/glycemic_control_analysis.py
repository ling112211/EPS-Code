from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def read_excel(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()
    return df


def pick_first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = {str(c).strip(): str(c).strip() for c in df.columns}
    for c in candidates:
        if c in cols:
            return cols[c]
    return None


def pick_case_insensitive(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    mapping = {str(c).strip().lower(): str(c).strip() for c in df.columns}
    for c in candidates:
        key = str(c).strip().lower()
        if key in mapping:
            return mapping[key]
    return None


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def clean_numeric_keepna(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def t_crit_975(df: float) -> float:
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except Exception:
        return 1.959963984540054


def mean_ci_95(series: pd.Series) -> Tuple[int, float, float, float, float]:
    s = pd.Series(series).dropna().astype(float)
    n = int(len(s))
    if n == 0:
        return 0, np.nan, np.nan, np.nan, np.nan
    m = float(s.mean())
    sd = float(s.std(ddof=1)) if n > 1 else 0.0
    if n <= 1:
        return n, m, sd, np.nan, np.nan
    se = sd / math.sqrt(n)
    tcrit = t_crit_975(n - 1)
    low = m - tcrit * se
    high = m + tcrit * se
    return n, m, sd, float(low), float(high)


def welch_ttest_pvalue(x: pd.Series, y: pd.Series) -> float:
    x = pd.Series(x).dropna().astype(float).values
    y = pd.Series(y).dropna().astype(float).values
    if len(x) < 2 or len(y) < 2:
        return np.nan
    try:
        from scipy import stats
        _, p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
        return float(p)
    except Exception:
        return np.nan


def welch_mean_diff_ci(x_eps: pd.Series, x_hum: pd.Series) -> Tuple[float, float, float]:
    x = pd.Series(x_eps).dropna().astype(float).values
    y = pd.Series(x_hum).dropna().astype(float).values
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, np.nan

    m1, m2 = float(np.mean(x)), float(np.mean(y))
    s1, s2 = float(np.std(x, ddof=1)), float(np.std(y, ddof=1))
    diff = m1 - m2

    se2 = (s1 * s1) / n1 + (s2 * s2) / n2
    if se2 <= 0:
        return float(diff), np.nan, np.nan

    num = se2 * se2
    den = ((s1 * s1 / n1) ** 2) / (n1 - 1) + ((s2 * s2 / n2) ** 2) / (n2 - 1)
    df = num / den if den > 0 else (n1 + n2 - 2)

    tcrit = t_crit_975(df)
    se = math.sqrt(se2)
    low = diff - tcrit * se
    high = diff + tcrit * se
    return float(diff), float(low), float(high)


def format_p(p: object) -> str:
    if p is None:
        return r"$P$ = NA"
    if isinstance(p, str) and p.strip().startswith("<"):
        thr = p.strip()[1:]
        return rf"$P$ < {thr}"
    try:
        pv = float(p)
    except Exception:
        return r"$P$ = NA"
    if np.isnan(pv):
        return r"$P$ = NA"
    if pv < 0.001:
        return r"$P$ < 0.001"
    return rf"$P$ = {pv:.3f}"


def scalar_to_float(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else np.nan


def read_results_table(path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=None)
    df = sheets["Results"] if "Results" in sheets else next(iter(sheets.values()))
    df.columns = df.columns.astype(str).str.strip()

    metric_col = "Characteristic" if "Characteristic" in df.columns else "Metric"
    if metric_col not in df.columns:
        raise ValueError(f"Cannot find metric column in {path}. Columns={list(df.columns)}")
    df[metric_col] = df[metric_col].astype(str).str.strip()
    return df.rename(columns={metric_col: "Metric"})


def get_result_row(df: pd.DataFrame, metric: str) -> pd.Series:
    exact = df.loc[df["Metric"].str.strip() == metric]
    if not exact.empty:
        return exact.iloc[0]

    fuzzy = df.loc[df["Metric"].str.contains(re.escape(metric), case=False, na=False)]
    if not fuzzy.empty:
        return fuzzy.iloc[0]

    raise ValueError(f"Cannot find metric: {metric}")


def get_p_from_row(row: pd.Series) -> object:
    for col in ["P value", "P_value", "p_value", "p-value", "pval", "p", "P"]:
        if col not in row.index:
            continue
        value = row[col]
        if pd.isna(value):
            continue
        text = str(value).strip()
        compact = (
            text.replace(" ", "")
            .replace("P=", "")
            .replace("p=", "")
            .replace("P<", "<")
            .replace("p<", "<")
        )
        if compact.startswith("<"):
            match = re.search(r"<([0-9]*\.?[0-9]+)", compact)
            return f"<{match.group(1)}" if match else "<0.001"
        match = re.search(r"([0-9]*\.?[0-9]+)", compact)
        if match:
            return float(match.group(1))
    return np.nan


def parse_mean_ci(cell: object) -> Tuple[float, float, float]:
    if pd.isna(cell):
        return np.nan, np.nan, np.nan

    text = str(cell).strip().replace("−", "-")
    match = re.search(
        r"^\s*([+-]?\d*\.?\d+)\s*\(\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*\)\s*$",
        text,
    )
    if match:
        return tuple(float(match.group(i)) for i in (1, 2, 3))

    number = re.search(r"([+-]?\d*\.?\d+)", text)
    if number:
        mean = float(number.group(1))
        return mean, mean, mean

    return np.nan, np.nan, np.nan


def apply_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.85,
            "axes.unicode_minus": False,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.tick_params(axis="both", labelsize=7.8, width=0.8, length=3, pad=2)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.grid(True, color="#D7DEE3", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)


def set_limits(ax: plt.Axes, lows: Sequence[float], highs: Sequence[float]) -> Tuple[float, float]:
    finite_lows = [float(v) for v in lows if np.isfinite(v)]
    finite_highs = [float(v) for v in highs if np.isfinite(v)]
    if not finite_lows or not finite_highs:
        return ax.get_ylim()

    lo = min(0.0, min(finite_lows))
    hi = max(finite_highs)
    span = max(hi - lo, abs(hi), 1.0)
    lower = lo - 0.08 * span
    upper = hi + 0.38 * span
    ax.set_ylim(lower, upper)
    return lower, upper


def asym_yerr(means: Sequence[float], lows: Sequence[float], highs: Sequence[float]) -> np.ndarray:
    low_err = [max(0.0, mean - low) for mean, low in zip(means, lows)]
    high_err = [max(0.0, high - mean) for mean, high in zip(means, highs)]
    return np.asarray([low_err, high_err])


def add_p_bracket(ax: plt.Axes, p_label: str, y: float, h: float) -> None:
    x0, x1 = 0, 1
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], color="#252525", lw=0.75)
    ax.text(
        0.5,
        y + h * 1.45,
        p_label,
        ha="center",
        va="bottom",
        fontsize=7.8,
        color="#252525",
    )


def format_arm_tick(label: str, n: object) -> str:
    n_float = scalar_to_float(n)
    if np.isfinite(n_float):
        return f"{label}\nn={int(n_float)}"
    return label


def fpg_reduction_from_frame(df: pd.DataFrame) -> pd.Series:
    baseline_col = pick_first_existing(
        df,
        [
            "baseline_fpg_mmol",
            "baseline_fpg",
            "FPG0",
            "Baseline fasting glucose",
            "Fasting glucose",
            "入营空腹",
        ],
    )
    endpoint_col = pick_first_existing(
        df,
        [
            "endpoint_fpg_mmol",
            "endpoint_fpg",
            "FPG1",
            "Endpoint fasting glucose",
            "结营空腹",
        ],
    )
    if baseline_col is None or endpoint_col is None:
        raise ValueError("Fasting glucose baseline/endline columns not found.")
    return clean_numeric_keepna(df[baseline_col]) - clean_numeric_keepna(df[endpoint_col])


def plot_glycemic_bars(
    out_dir: Path,
    human_mean_red: float,
    human_low_red: float,
    human_high_red: float,
    eps_mean_red: float,
    eps_low_red: float,
    eps_high_red: float,
    human_mean_pct: float,
    human_low_pct: float,
    human_high_pct: float,
    eps_mean_pct: float,
    eps_low_pct: float,
    eps_high_pct: float,
    p_red: float,
    p_pct: float,
    human_n: object = np.nan,
    eps_n: object = np.nan,
    individual_reduction: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    apply_plot_style()

    arm_order = ("Human", "EPS-human")
    arm_colors = {
        "Human": "#AAB7C0",
        "EPS-human": "#0072B2",
    }
    arm_edges = {
        "Human": "#6E7C86",
        "EPS-human": "#004C79",
    }
    xticklabels = [
        format_arm_tick("Human", human_n),
        format_arm_tick("EPS-human", eps_n),
    ]
    x = np.arange(len(arm_order))

    if individual_reduction is None:
        raise ValueError(
            "Fig. 3b requires participant-level fasting glucose reductions for "
            "the individual-outcomes panel."
        )

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.65), dpi=400)
    fig.patch.set_facecolor("white")

    for ax in axes:
        style_axis(ax)

    def draw_panel(
        ax: plt.Axes,
        means: Sequence[float],
        lows: Sequence[float],
        highs: Sequence[float],
        title: str,
        ylabel: str,
        pval: object,
    ) -> None:
        ax.bar(
            x,
            means,
            width=0.58,
            color=[arm_colors[a] for a in arm_order],
            edgecolor=[arm_edges[a] for a in arm_order],
            linewidth=0.65,
            alpha=0.70,
            zorder=2,
        )
        ax.errorbar(
            x,
            means,
            yerr=asym_yerr(means, lows, highs),
            fmt="o",
            markersize=3.4,
            markerfacecolor="#FFFFFF",
            markeredgecolor="#1F1F1F",
            ecolor="#1F1F1F",
            elinewidth=0.95,
            capsize=3,
            capthick=0.95,
            zorder=4,
        )

        lower, upper = set_limits(ax, lows, highs)
        span = upper - lower
        ax.axhline(0, color="#4B4B4B", linewidth=0.75, zorder=1)

        bracket_y = max(float(v) for v in highs if np.isfinite(v)) + 0.15 * span
        bracket_h = 0.025 * span
        add_p_bracket(ax, format_p(pval), bracket_y, bracket_h)

        ax.set_title(title, loc="center", fontsize=9.0, fontweight="normal", pad=7)
        ax.set_ylabel(ylabel, fontsize=8.0)
        ax.set_xticks(x, xticklabels)
        ax.set_xlim(-0.55, 1.55)

    draw_panel(
        axes[0],
        means=[human_mean_red, eps_mean_red],
        lows=[human_low_red, eps_low_red],
        highs=[human_high_red, eps_high_red],
        title="Fasting glucose reduction",
        ylabel="Mean (95% CI), mmol/L",
        pval=p_red,
    )
    draw_panel(
        axes[1],
        means=[human_mean_pct, eps_mean_pct],
        lows=[human_low_pct, eps_low_pct],
        highs=[human_high_pct, eps_high_pct],
        title="Percent fasting glucose reduction",
        ylabel="Mean (95% CI), %",
        pval=p_pct,
    )

    ax = axes[2]
    rng = np.random.default_rng(20260521)
    all_values = np.concatenate(
        [
            np.asarray(individual_reduction[arm], dtype=float)[
                np.isfinite(np.asarray(individual_reduction[arm], dtype=float))
            ]
            for arm in arm_order
        ]
    )
    if all_values.size == 0:
        raise ValueError("No valid participant-level fasting glucose reductions were supplied.")
    y_min = float(np.min(all_values))
    y_max = float(np.max(all_values))
    y_span = max(y_max - y_min, 1.0)

    for idx, arm in enumerate(arm_order):
        values = np.asarray(individual_reduction[arm], dtype=float)
        values = values[np.isfinite(values)]
        jitter = rng.uniform(-0.09, 0.09, size=len(values))
        ax.scatter(
            np.full(len(values), idx, dtype=float) + jitter,
            values,
            s=17,
            color=arm_colors[arm],
            edgecolor=arm_edges[arm],
            linewidth=0.45,
            alpha=0.76,
            zorder=3,
        )
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        ax.plot([idx - 0.18, idx + 0.18], [median, median], color="#1F1F1F", lw=1.15)
        ax.plot([idx, idx], [q1, q3], color="#1F1F1F", lw=0.95)

    style_axis(ax)
    ax.axhline(0, color="#4B4B4B", linewidth=0.75, zorder=1)
    ax.set_title("Individual fasting glucose reduction", fontsize=9.0, pad=7)
    ax.set_ylabel("Reduction, mmol/L", fontsize=8.0)
    ax.set_xticks([0, 1], xticklabels)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(y_min - 0.16 * y_span, y_max + 0.18 * y_span)

    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.20, top=0.91, wspace=0.48)

    out_paths = [
        out_dir / "glycemic_fpg_reduction_bars.png",
        out_dir / "glycemic_fpg_reduction_bars.pdf",
        out_dir / "glycemic_fpg_reduction_bars_nm_style.png",
        out_dir / "glycemic_fpg_reduction_bars_nm_style.pdf",
    ]
    ensure_parent_dir(out_paths[0])
    for path in out_paths:
        if path.suffix.lower() == ".png":
            fig.savefig(path, bbox_inches="tight", dpi=600)
        else:
            fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def parse_participant_n(row: pd.Series, arm_col: str) -> float:
    value = row.get(arm_col, np.nan)
    if pd.isna(value):
        return np.nan
    match = re.search(r"(\d+)", str(value))
    return float(match.group(1)) if match else scalar_to_float(value)


def arm_mean_ci(row: pd.Series, arm: str) -> Tuple[float, float, float]:
    if arm in row.index:
        return parse_mean_ci(row[arm])

    prefix = "Human" if arm == "Human" else "EPS"
    return (
        scalar_to_float(row.get(f"{prefix}_Mean", np.nan)),
        scalar_to_float(row.get(f"{prefix}_Mean_CI_low", np.nan)),
        scalar_to_float(row.get(f"{prefix}_Mean_CI_high", np.nan)),
    )


def plot_glycemic_bars_from_results(
    out_dir: Path,
    results: pd.DataFrame,
    individual_reduction: Dict[str, np.ndarray],
) -> None:
    row_red = get_result_row(results, "Fasting glucose reduction")
    row_pct = get_result_row(results, "Fasting glucose reduction ratio (%)")

    n_h = row_red.get("Human_N", np.nan)
    n_e = row_red.get("EPS_N", np.nan)
    try:
        row_n = get_result_row(results, "No. of participants")
        n_h = parse_participant_n(row_n, "Human")
        n_e = parse_participant_n(row_n, "EPS-human")
    except ValueError:
        pass

    human_red = arm_mean_ci(row_red, "Human")
    eps_red = arm_mean_ci(row_red, "EPS-human")
    human_pct = arm_mean_ci(row_pct, "Human")
    eps_pct = arm_mean_ci(row_pct, "EPS-human")

    plot_glycemic_bars(
        out_dir=out_dir,
        human_mean_red=human_red[0],
        human_low_red=human_red[1],
        human_high_red=human_red[2],
        eps_mean_red=eps_red[0],
        eps_low_red=eps_red[1],
        eps_high_red=eps_red[2],
        human_mean_pct=human_pct[0],
        human_low_pct=human_pct[1],
        human_high_pct=human_pct[2],
        eps_mean_pct=eps_pct[0],
        eps_low_pct=eps_pct[1],
        eps_high_pct=eps_pct[2],
        p_red=get_p_from_row(row_red),
        p_pct=get_p_from_row(row_pct),
        human_n=n_h,
        eps_n=n_e,
        individual_reduction=individual_reduction,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fasting glucose outcomes (Fig. 3b).")
    p.add_argument("--gly_human", type=str, default=None, help="Path to Human glycemic-control Excel.")
    p.add_argument("--gly_eps", type=str, default=None, help="Path to EPS-human glycemic-control Excel.")
    p.add_argument(
        "--summary-xlsx",
        "--summary_xlsx",
        dest="summary_xlsx",
        type=str,
        default=None,
        help="Optional final summary workbook with a Results sheet, matching the manuscript plotting workflow.",
    )
    p.add_argument("--out_dir", type=str, default="outputs/clinical_trial", help="Output directory.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    if args.summary_xlsx is not None:
        if args.gly_human is None or args.gly_eps is None:
            raise SystemExit(
                "Fig. 3b now includes an individual reduction panel. With --summary-xlsx, "
                "also provide --gly_human and --gly_eps for participant-level values."
            )
        results = read_results_table(Path(args.summary_xlsx))
        individual_values = {
            "Human": fpg_reduction_from_frame(read_excel(Path(args.gly_human))).dropna().to_numpy(dtype=float),
            "EPS-human": fpg_reduction_from_frame(read_excel(Path(args.gly_eps))).dropna().to_numpy(dtype=float),
        }
        plot_glycemic_bars_from_results(out_dir, results, individual_values)
        return

    if args.gly_human is None or args.gly_eps is None:
        raise SystemExit("Missing input paths. Provide --gly_human and --gly_eps, or use --summary-xlsx.")

    df_h = read_excel(Path(args.gly_human))
    df_e = read_excel(Path(args.gly_eps))

    baseline_fpg_candidates = ["baseline_fpg_mmol", "baseline_fpg", "FPG0", "Baseline fasting glucose", "Fasting glucose", "入营空腹"]
    end_fpg_candidates = ["endpoint_fpg_mmol", "endpoint_fpg", "FPG1", "Endpoint fasting glucose", "结营空腹"]

    col_fpg0_h = pick_first_existing(df_h, baseline_fpg_candidates)
    col_fpg1_h = pick_first_existing(df_h, end_fpg_candidates)
    col_fpg0_e = pick_first_existing(df_e, baseline_fpg_candidates)
    col_fpg1_e = pick_first_existing(df_e, end_fpg_candidates)

    if col_fpg0_h is None or col_fpg1_h is None or col_fpg0_e is None or col_fpg1_e is None:
        raise ValueError(
            "Fasting glucose columns not found in both arms. "
            f"Baseline_candidates={baseline_fpg_candidates}, End_candidates={end_fpg_candidates}. "
            f"Human_found=({col_fpg0_h},{col_fpg1_h}), EPS_found=({col_fpg0_e},{col_fpg1_e})"
        )

    df_h = df_h.copy()
    df_e = df_e.copy()

    df_h["fpg0"] = clean_numeric_keepna(df_h[col_fpg0_h])
    df_h["fpg1"] = clean_numeric_keepna(df_h[col_fpg1_h])
    df_e["fpg0"] = clean_numeric_keepna(df_e[col_fpg0_e])
    df_e["fpg1"] = clean_numeric_keepna(df_e[col_fpg1_e])

    df_h["fpg_red"] = df_h["fpg0"] - df_h["fpg1"]
    df_e["fpg_red"] = df_e["fpg0"] - df_e["fpg1"]

    df_h["fpg_red_pct"] = np.where(df_h["fpg0"] > 0, df_h["fpg_red"] / df_h["fpg0"] * 100.0, np.nan)
    df_e["fpg_red_pct"] = np.where(df_e["fpg0"] > 0, df_e["fpg_red"] / df_e["fpg0"] * 100.0, np.nan)

    # Arm summaries
    n_h_red, m_h_red, sd_h_red, lo_h_red, hi_h_red = mean_ci_95(df_h["fpg_red"])
    n_e_red, m_e_red, sd_e_red, lo_e_red, hi_e_red = mean_ci_95(df_e["fpg_red"])

    n_h_pct, m_h_pct, sd_h_pct, lo_h_pct, hi_h_pct = mean_ci_95(df_h["fpg_red_pct"])
    n_e_pct, m_e_pct, sd_e_pct, lo_e_pct, hi_e_pct = mean_ci_95(df_e["fpg_red_pct"])

    # Between-arm Welch
    p_red = welch_ttest_pvalue(df_e["fpg_red"], df_h["fpg_red"])
    diff_red, diff_red_lo, diff_red_hi = welch_mean_diff_ci(df_e["fpg_red"], df_h["fpg_red"])

    p_pct = welch_ttest_pvalue(df_e["fpg_red_pct"], df_h["fpg_red_pct"])
    diff_pct, diff_pct_lo, diff_pct_hi = welch_mean_diff_ci(df_e["fpg_red_pct"], df_h["fpg_red_pct"])

    results = pd.DataFrame(
        [
            {
                "Metric": "Fasting glucose reduction",
                "Human_N": n_h_red,
                "Human_Mean": m_h_red,
                "Human_SD": sd_h_red,
                "Human_Mean_CI_low": lo_h_red,
                "Human_Mean_CI_high": hi_h_red,
                "EPS_N": n_e_red,
                "EPS_Mean": m_e_red,
                "EPS_SD": sd_e_red,
                "EPS_Mean_CI_low": lo_e_red,
                "EPS_Mean_CI_high": hi_e_red,
                "MeanDiff": diff_red,
                "MeanDiff_CI_low": diff_red_lo,
                "MeanDiff_CI_high": diff_red_hi,
                "P_value": p_red,
                "p_value": p_red,
            },
            {
                "Metric": "Fasting glucose reduction ratio (%)",
                "Human_N": n_h_pct,
                "Human_Mean": m_h_pct,
                "Human_SD": sd_h_pct,
                "Human_Mean_CI_low": lo_h_pct,
                "Human_Mean_CI_high": hi_h_pct,
                "EPS_N": n_e_pct,
                "EPS_Mean": m_e_pct,
                "EPS_SD": sd_e_pct,
                "EPS_Mean_CI_low": lo_e_pct,
                "EPS_Mean_CI_high": hi_e_pct,
                "MeanDiff": diff_pct,
                "MeanDiff_CI_low": diff_pct_lo,
                "MeanDiff_CI_high": diff_pct_hi,
                "P_value": p_pct,
                "p_value": p_pct,
            },
        ]
    )

    out_xlsx = out_dir / "glycemic_control_analysis_results.xlsx"
    ensure_parent_dir(out_xlsx)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        results.to_excel(writer, index=False, sheet_name="Results")

    # Plot (publication-style)
    plot_glycemic_bars(
        out_dir=out_dir,
        human_mean_red=m_h_red,
        human_low_red=lo_h_red,
        human_high_red=hi_h_red,
        eps_mean_red=m_e_red,
        eps_low_red=lo_e_red,
        eps_high_red=hi_e_red,
        human_mean_pct=m_h_pct,
        human_low_pct=lo_h_pct,
        human_high_pct=hi_h_pct,
        eps_mean_pct=m_e_pct,
        eps_low_pct=lo_e_pct,
        eps_high_pct=hi_e_pct,
        p_red=p_red,
        p_pct=p_pct,
        human_n=n_h_red,
        eps_n=n_e_red,
        individual_reduction={
            "Human": df_h["fpg_red"].dropna().to_numpy(dtype=float),
            "EPS-human": df_e["fpg_red"].dropna().to_numpy(dtype=float),
        },
    )


if __name__ == "__main__":
    main()
