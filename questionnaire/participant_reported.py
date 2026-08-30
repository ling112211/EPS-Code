import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import t as t_dist
from scipy.stats import ttest_ind


# =========================
# Plot style
# =========================
def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
    })


# =========================
# Column detection
# =========================
Q1_PATTERN = re.compile(r"^\s*(?:Q1\s*$|1\s*\.)")
Q2_15_PATTERN = re.compile(r"^\s*(?:Q(?:[2-9]|1[0-5])\s*$|(?:[2-9]|1[0-5])\s*\.)")


def normalize_colname(x: object) -> str:
    return str(x).strip()


def pick_question_cols(df: pd.DataFrame, pattern: re.Pattern) -> List[str]:
    cols = []
    for c in df.columns:
        s = normalize_colname(c)
        if pattern.match(s):
            cols.append(c)
    return cols


def qnum(colname: object) -> int:
    s = normalize_colname(colname)
    # Match "Q{n}" format first, then "{n}." legacy format
    m = re.match(r"^\s*Q(\d+)\s*$", s, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.match(r"^\s*(\d+)\s*\.", s)
    return int(m.group(1)) if m else 10**9


def find_single_col(df: pd.DataFrame, pattern: re.Pattern) -> str:
    cols = pick_question_cols(df, pattern)
    if len(cols) != 1:
        raise ValueError(f"Expected exactly one column matching pattern, got {len(cols)}: {cols}")
    return cols[0]


# =========================
# Parsing helpers
# =========================
def parse_yes_no(x: object) -> float:
    """
    Parse screening item (Q1).
    Returns 1.0 for "A" (yes), 0.0 for "B" (no), otherwise NaN.
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    m = re.match(r"^([ABab])\s*[\.\．、。:：\)\）\s]?", s)
    if m:
        return 1.0 if m.group(1).upper() == "A" else 0.0

    if "是" in s and "否" not in s:
        return 1.0
    if "否" in s and "是" not in s:
        return 0.0

    # Heuristic for common English yes/no tokens.
    sl = s.lower()
    if "yes" in sl and "no" not in sl:
        return 1.0
    if "no" in sl and "yes" not in sl:
        return 0.0
    return np.nan


def parse_likert_1_7(x: object) -> float:
    """
    Parse a 1–7 Likert response.
    Accepts raw numeric values or strings like "5", "5.XXX".
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x)
        return v if 1.0 <= v <= 7.0 else np.nan

    s = str(x).strip()
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)", s)
    if not m:
        return np.nan
    v = float(m.group(1))
    return v if 1.0 <= v <= 7.0 else np.nan


# =========================
# Stats helpers
# =========================
def mean_ci_t(x: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float, int, float]:
    """
    Mean and two-sided (1-alpha) CI using Student t distribution.
    Returns (mean, ci_low, ci_high, n, sd).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = int(x.size)
    if n < 2:
        return np.nan, np.nan, np.nan, n, np.nan

    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    se = sd / float(np.sqrt(n))

    tcrit = float(t_dist.ppf(1.0 - alpha / 2.0, df=n - 1))

    ci_low = mean - tcrit * se
    ci_high = mean + tcrit * se
    return mean, ci_low, ci_high, n, sd


def welch_t_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    """
    Two-sided Welch's t-test p-value.
    Requires SciPy so that the reported test uses the Welch t distribution.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if x.size < 2 or y.size < 2:
        return np.nan

    return float(ttest_ind(x, y, equal_var=False, nan_policy="omit").pvalue)


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """
    Holm step-down adjustment.
    Preserves NaNs: NaN inputs map to NaN outputs.
    """
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
        p = float(pv[idx])
        adj_p = (m - k) * p
        running_max = max(running_max, adj_p)
        adj[idx] = min(running_max, 1.0)

    out[mask] = adj
    return out


# =========================
# QC helpers
# =========================
def try_find_id_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["id", "ID", "Id", "participant_id", "user_id", "编号"]
    for c in df.columns:
        if normalize_colname(c) in candidates:
            return c
    return None


def try_find_time_col(df: pd.DataFrame) -> Optional[str]:
    """
    Tries to locate a completion-time column by common name patterns.
    """
    patterns = [
        r"(?i)\b(duration|time\s*spent|elapsed|completion\s*time)\b",
    ]
    for c in df.columns:
        name = normalize_colname(c)
        for pat in patterns:
            if re.search(pat, name):
                return c
    return None


def infer_time_seconds(values: pd.Series, colname: str) -> pd.Series:
    """
    Convert a time column to seconds using a conservative heuristic.
    If units are ambiguous, assumes seconds.
    """
    s = pd.to_numeric(values, errors="coerce")
    name = normalize_colname(colname).lower()

    if ("min" in name) or ("minute" in name):
        return s * 60.0
    if ("sec" in name) or ("second" in name):
        return s

    # Heuristic by magnitude: large medians are likely seconds.
    med = float(np.nanmedian(s.to_numpy(dtype=float))) if np.isfinite(np.nanmedian(s.to_numpy(dtype=float))) else np.nan
    if np.isnan(med):
        return s
    if med <= 30.0:
        # Could be minutes; keep conservative and treat as minutes.
        return s * 60.0
    return s


def drop_straightliners(df: pd.DataFrame, q_cols: List[str], min_answered: int = 8) -> Tuple[pd.DataFrame, int]:
    """
    Drops rows that select the same option for all answered items (Q2..Q15).
    Only considers rows with at least min_answered non-missing parsed responses.
    """
    parsed = df[q_cols].map(parse_likert_1_7)
    answered = parsed.notna().sum(axis=1)
    same = parsed.nunique(axis=1, dropna=True)
    mask_keep = ~((answered >= min_answered) & (same <= 1))
    removed = int((~mask_keep).sum())
    return df.loc[mask_keep].copy(), removed


def apply_time_filter(
    df: pd.DataFrame,
    time_col: str,
    min_time_s: float,
    max_time_s: float
) -> Tuple[pd.DataFrame, int]:
    """
    Drops rows outside [min_time_s, max_time_s] based on inferred seconds.
    """
    tsec = infer_time_seconds(df[time_col], time_col)
    mask_keep = (tsec >= min_time_s) & (tsec <= max_time_s)
    removed = int((~mask_keep).sum())
    return df.loc[mask_keep].copy(), removed


def likert_invalid_reason(x: object) -> Optional[str]:
    """Classify missing, unparseable, or out-of-range Q2-Q15 responses."""
    if pd.isna(x):
        return "missing"
    if isinstance(x, (int, float, np.integer, np.floating)):
        value = float(x)
    else:
        match = re.match(r"^\s*(-?\d+(?:\.\d+)?)", str(x).strip())
        if not match:
            return "unparseable"
        value = float(match.group(1))
    if not 1.0 <= value <= 7.0:
        return "out_of_range"
    return None


def drop_invalid_questionnaires(
    df: pd.DataFrame,
    q_cols: List[str],
    group_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude an entire questionnaire if any Q2-Q15 response is invalid."""
    invalid_indices = set()
    audit_rows = []
    for source_row_index, row in df.iterrows():
        invalid_items = []
        invalid_reasons = []
        for col in q_cols:
            reason = likert_invalid_reason(row[col])
            if reason is not None:
                invalid_items.append(f"Q{qnum(col)}")
                invalid_reasons.append(reason)
        if invalid_items:
            invalid_indices.add(source_row_index)
            audit_rows.append({
                "group": group_name,
                "source_row_index": source_row_index,
                "invalid_item_count": len(invalid_items),
                "invalid_items": ",".join(invalid_items),
                "invalid_reasons": ",".join(invalid_reasons),
            })

    audit = pd.DataFrame(
        audit_rows,
        columns=[
            "group",
            "source_row_index",
            "invalid_item_count",
            "invalid_items",
            "invalid_reasons",
        ],
    )
    return df.loc[~df.index.isin(invalid_indices)].copy(), audit


# =========================
# Core analysis
# =========================
def summarize_group(df: pd.DataFrame, q_cols: List[str], group_name: str) -> pd.DataFrame:
    rows = []
    for c in q_cols:
        vals = df[c].map(parse_likert_1_7).to_numpy(dtype=float)
        mean, lo, hi, n, sd = mean_ci_t(vals, alpha=0.05)
        rows.append({
            "group": group_name,
            "question": qnum(c),
            "col": normalize_colname(c),
            "n": n,
            "mean": mean,
            "sd": sd,
            "ci_low": lo,
            "ci_high": hi,
        })
    return pd.DataFrame(rows).sort_values("question").reset_index(drop=True)


def compute_item_pvalues(
    df_h: pd.DataFrame, df_e: pd.DataFrame,
    q_cols_h: List[str], q_cols_e: List[str]
) -> np.ndarray:
    pvals = []
    for ch, ce in zip(q_cols_h, q_cols_e):
        x = df_h[ch].map(parse_likert_1_7).to_numpy(dtype=float)
        y = df_e[ce].map(parse_likert_1_7).to_numpy(dtype=float)
        pvals.append(welch_t_pvalue(x, y))
    return np.asarray(pvals, dtype=float)


def make_wide_table(sum_h: pd.DataFrame, sum_e: pd.DataFrame, pvals: np.ndarray) -> pd.DataFrame:
    p_holm = holm_adjust(pvals)

    wide = (
        sum_h[["question", "n", "mean", "ci_low", "ci_high"]]
        .rename(columns={
            "n": "n_human",
            "mean": "mean_human",
            "ci_low": "ci_low_human",
            "ci_high": "ci_high_human",
        })
        .merge(
            sum_e[["question", "n", "mean", "ci_low", "ci_high"]]
            .rename(columns={
                "n": "n_eps",
                "mean": "mean_eps",
                "ci_low": "ci_low_eps",
                "ci_high": "ci_high_eps",
            }),
            on="question",
            how="inner",
        )
    )
    wide["p_value_welch"] = pvals
    wide["p_holm_14tests"] = p_holm
    return wide


# =========================
# Plotting
# =========================
DEFAULT_LABELS: Dict[int, str] = {
    2:  "Overall satisfaction",
    3:  "Pleasant interaction",
    4:  "Happiness after feedback",
    5:  "Calm & emotionally stable",
    6:  "Steady tone under pressure",
    7:  "Not easily triggered",
    8:  "Low emotional swings",
    9:  "Not anxious",
    10: "Tone consistency",
    11: "Often agrees",
    12: "Rarely disagrees",
    13: "Emphasizes agreement",
    14: "Frequent praise",
    15: "Highlights strengths",
}

DOMAINS: Dict[str, List[int]] = {
    "Satisfaction and interaction": [2, 3, 4],
    "Emotional stability": [5, 6, 7, 8, 9, 10],
    "Agreement and affirmation": [11, 12, 13, 14, 15],
}


def radar_plot_with_ci(
    sum_h: pd.DataFrame,
    sum_e: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
    labels: Dict[int, str],
    group_h_name: str,
    group_e_name: str,
) -> None:
    dim_qnums = list(range(2, 16))
    cat_labels = [labels[i] for i in dim_qnums]

    def grab(series_name: str, q: int) -> float:
        v = sum_h.loc[sum_h["question"] == q, series_name].values
        if v.size != 1:
            return np.nan
        return float(v[0])

    def grab_e(series_name: str, q: int) -> float:
        v = sum_e.loc[sum_e["question"] == q, series_name].values
        if v.size != 1:
            return np.nan
        return float(v[0])

    h_means = [grab("mean", q) for q in dim_qnums]
    h_lo = [grab("ci_low", q) for q in dim_qnums]
    h_hi = [grab("ci_high", q) for q in dim_qnums]

    e_means = [grab_e("mean", q) for q in dim_qnums]
    e_lo = [grab_e("ci_low", q) for q in dim_qnums]
    e_hi = [grab_e("ci_high", q) for q in dim_qnums]

    def close(arr: List[float]) -> List[float]:
        return arr + [arr[0]]

    N = len(dim_qnums)
    angles = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False).tolist()
    angles_c = angles + [angles[0]]

    h_means_c = close(h_means)
    e_means_c = close(e_means)

    fig = plt.figure(figsize=(5.95, 5.45), dpi=300)
    ax = plt.subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)

    ax.set_ylim(1, 7)
    ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_yticklabels([])

    grid_color = "#D7DEE3"
    ax.yaxis.grid(True, color=grid_color, linewidth=0.7, alpha=0.82)
    ax.xaxis.grid(True, color=grid_color, linewidth=0.7, alpha=0.82)
    ax.spines["polar"].set_linewidth(0.85)
    ax.spines["polar"].set_color("#1F1F1F")

    ax.set_xticks(angles)
    ax.set_xticklabels(cat_labels, fontfamily="Times New Roman", fontsize=7.5)
    ax.tick_params(axis="x", pad=9)
    for tick_label, angle in zip(ax.get_xticklabels(), angles):
        display_angle = (np.pi / 2 - angle + np.pi) % (2 * np.pi) - np.pi
        if abs(abs(display_angle) - np.pi / 2) < 1e-9:
            tick_label.set_horizontalalignment("center")
        elif -np.pi / 2 < display_angle < np.pi / 2:
            tick_label.set_horizontalalignment("left")
        else:
            tick_label.set_horizontalalignment("right")

    col_h = "#6E7C86"
    col_h_fill = "#AAB7C0"
    col_e = "#0072B2"
    col_e_fill = "#0072B2"

    ax.plot(angles_c, h_means_c, color=col_h, linewidth=1.7, label=group_h_name)
    ax.fill(angles_c, h_means_c, color=col_h_fill, alpha=0.22)

    ax.plot(angles_c, e_means_c, color=col_e, linewidth=1.8, label=group_e_name)
    ax.fill(angles_c, e_means_c, color=col_e_fill, alpha=0.12)

    ax.scatter(angles, h_means, color=col_h, s=13, zorder=3)
    ax.scatter(angles, e_means, color=col_e, s=13, zorder=3)

    def draw_errorbars(
        angles_: List[float],
        lo_: List[float],
        hi_: List[float],
        color: str,
        cap: float = 0.025,
        lw: float = 0.95,
        alpha: float = 0.95
    ) -> None:
        for th, l, u in zip(angles_, lo_, hi_):
            if np.isnan(l) or np.isnan(u):
                continue
            ax.plot([th, th], [l, u], color=color, linewidth=lw, alpha=alpha, zorder=2)
            ax.plot([th - cap, th + cap], [l, l], color=color, linewidth=lw, alpha=alpha, zorder=2)
            ax.plot([th - cap, th + cap], [u, u], color=color, linewidth=lw, alpha=alpha, zorder=2)

    draw_errorbars(angles, h_lo, h_hi, col_h)
    draw_errorbars(angles, e_lo, e_hi, col_e)

    ax.legend(
        handles=[
            Line2D([0], [0], color=col_e, lw=1.8, label=group_e_name),
            Line2D([0], [0], color=col_h, lw=1.7, label=group_h_name),
        ],
        loc="upper left",
        bbox_to_anchor=(1.08, 1.06),
        frameon=False,
        fontsize=8.0,
    )

    fig.subplots_adjust(left=0.08, right=0.82, bottom=0.06, top=0.96)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=600)
    plt.close(fig)


def valid_scores(df: pd.DataFrame, q_cols: List[str], q: int) -> np.ndarray:
    col = next(col for col in q_cols if qnum(col) == q)
    values = df[col].map(parse_likert_1_7).to_numpy(dtype=float)
    return values[~np.isnan(values)]


def diff_ci(h_values: np.ndarray, e_values: np.ndarray) -> Tuple[float, float, float]:
    mean_h = float(h_values.mean())
    mean_e = float(e_values.mean())
    var_h = float(h_values.var(ddof=1))
    var_e = float(e_values.var(ddof=1))
    n_h = h_values.size
    n_e = e_values.size
    difference = mean_e - mean_h
    se = math.sqrt(var_h / n_h + var_e / n_e)
    if se == 0:
        return difference, difference, difference

    numerator = (var_h / n_h + var_e / n_e) ** 2
    denominator = ((var_h / n_h) ** 2 / (n_h - 1)) + ((var_e / n_e) ** 2 / (n_e - 1))
    degrees_freedom = numerator / denominator if denominator else min(n_h, n_e) - 1
    critical = float(t_dist.ppf(0.975, df=degrees_freedom))
    return difference, difference - critical * se, difference + critical * se


def build_item_difference_table(
    df_h: pd.DataFrame,
    df_e: pd.DataFrame,
    q_cols_h: List[str],
    q_cols_e: List[str],
) -> pd.DataFrame:
    domain_by_q = {question: domain for domain, questions in DOMAINS.items() for question in questions}
    rows = []
    for question in range(2, 16):
        difference, low, high = diff_ci(
            valid_scores(df_h, q_cols_h, question),
            valid_scores(df_e, q_cols_e, question),
        )
        rows.append(
            {
                "question": question,
                "label": DEFAULT_LABELS[question],
                "domain": domain_by_q[question],
                "mean_difference": difference,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def build_domain_distribution_table(
    df_h: pd.DataFrame,
    df_e: pd.DataFrame,
    q_cols_h: List[str],
    q_cols_e: List[str],
    group_h_name: str,
    group_e_name: str,
) -> pd.DataFrame:
    rows = []
    for domain, questions in DOMAINS.items():
        for group, frame, q_cols in [
            (group_h_name, df_h, q_cols_h),
            (group_e_name, df_e, q_cols_e),
        ]:
            scores = np.concatenate([valid_scores(frame, q_cols, question) for question in questions])
            categories = {
                "1-3": (scores >= 1) & (scores <= 3),
                "4": scores == 4,
                "5": scores == 5,
                "6-7": (scores >= 6) & (scores <= 7),
            }
            for category, mask in categories.items():
                rows.append(
                    {
                        "domain": domain,
                        "group": group,
                        "category": category,
                        "percent": float(np.mean(mask) * 100),
                        "n_item_responses": int(len(scores)),
                    }
                )
    return pd.DataFrame(rows)


def plot_item_differences(
    diff_df: pd.DataFrame,
    holm_pvalues: np.ndarray,
    out_pdf: Path,
    out_png: Path,
) -> None:
    plot_df = diff_df.iloc[::-1].reset_index(drop=True)
    domain_colors = {
        "Satisfaction and interaction": "#0072B2",
        "Emotional stability": "#3A8F7B",
        "Agreement and affirmation": "#B05A2A",
    }
    fig, ax = plt.subplots(figsize=(5.55, 4.35), dpi=300)
    y = np.arange(len(plot_df))
    for idx, row in plot_df.iterrows():
        ax.errorbar(
            row["mean_difference"],
            idx,
            xerr=np.asarray(
                [
                    [row["mean_difference"] - row["ci_low"]],
                    [row["ci_high"] - row["mean_difference"]],
                ]
            ),
            fmt="o",
            markersize=3.8,
            color=domain_colors[row["domain"]],
            ecolor="#252525",
            elinewidth=0.85,
            capsize=2.5,
            capthick=0.85,
            zorder=3,
        )
    ax.axvline(0, color="#4B4B4B", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"Q{int(q)} {label}" for q, label in zip(plot_df["question"], plot_df["label"])],
        fontsize=6.5,
    )
    ax.set_xlabel("Mean difference (EPS-human - Human)", fontsize=8.0)
    ax.set_title("Item-level differences", fontsize=9.0, pad=6)
    ax.set_ylim(-0.5, len(plot_df) - 0.5)
    ax.set_xlim(0, max(2.05, float(plot_df["ci_high"].max() + 0.08)))
    ax.xaxis.grid(True, color="#D7DEE3", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=7.2, length=2.5, width=0.7)
    ax.tick_params(axis="y", length=0, pad=2.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)
    if np.all(np.asarray(holm_pvalues, dtype=float) < 0.001):
        pvalue_text = "All Holm-adjusted $P < 0.001$"
    else:
        pvalue_text = "Holm-adjusted $P$ values reported in table"
    ax.text(0.03, 0.97, pvalue_text, transform=ax.transAxes, ha="left", va="top", fontsize=7.0, color="#252525")
    fig.subplots_adjust(left=0.36, right=0.99, bottom=0.13, top=0.92)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=600)
    plt.close(fig)


def plot_domain_distributions(
    dist_df: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
    group_h_name: str,
    group_e_name: str,
) -> None:
    category_order = ["1-3", "4", "5", "6-7"]
    category_colors = {"1-3": "#D6DDE1", "4": "#EFE0A2", "5": "#8FC1D4", "6-7": "#0072B2"}
    domain_short = {
        "Satisfaction and interaction": "Satisfaction",
        "Emotional stability": "Emotional\nstability",
        "Agreement and affirmation": "Agreement/\naffirmation",
    }
    rows = []
    for domain in DOMAINS:
        rows.extend([(domain, group_h_name), (domain, group_e_name)])

    fig, ax = plt.subplots(figsize=(5.55, 3.35), dpi=300)
    y_positions = np.arange(len(rows))[::-1]
    for ypos, (domain, group) in zip(y_positions, rows):
        left = 0.0
        subset = dist_df.loc[(dist_df["domain"] == domain) & (dist_df["group"] == group)]
        for category in category_order:
            percent = float(subset.loc[subset["category"] == category, "percent"].iloc[0])
            ax.barh(ypos, percent, left=left, height=0.62, color=category_colors[category], edgecolor="white", linewidth=0.45)
            left += percent

    ax.set_xlim(0, 100)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{domain_short[domain]}\n{group}" for domain, group in rows], fontsize=6.6)
    ax.set_xlabel("Responses, %", fontsize=8.0)
    ax.set_title("Domain response distributions", fontsize=9.0, pad=6)
    ax.xaxis.grid(True, color="#D7DEE3", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=7.2, length=2.5, width=0.7)
    ax.tick_params(axis="y", length=0, pad=2.0)
    ax.legend(
        handles=[Patch(facecolor=category_colors[category], edgecolor="white", label=category) for category in category_order],
        title="Score",
        loc="lower center",
        bbox_to_anchor=(0.52, -0.31),
        frameon=False,
        ncol=4,
        fontsize=7.0,
        title_fontsize=7.1,
        handlelength=1.0,
        columnspacing=0.9,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)
    fig.subplots_adjust(left=0.24, right=0.99, bottom=0.27, top=0.90)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=600)
    plt.close(fig)


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Participant-reported experience (Fig. 3a-c) analysis and plots.")
    p.add_argument("--human-xlsx", type=str, required=True, help="Path to Human summary.xlsx")
    p.add_argument("--eps-xlsx", type=str, required=True, help="Path to EPS-human clean_responses.xlsx")
    p.add_argument("--sheet", type=str, default="0", help="Excel sheet name or index (default: 0)")

    p.add_argument("--outdir", type=str, default="outputs/questionnaire", help="Output directory")
    p.add_argument("--prefix", type=str, default="phase2", help="Output filename prefix")

    p.add_argument("--filter-used-only", action="store_true", help="Keep only respondents with Q1 == yes")
    p.add_argument("--no-filter-used-only", dest="filter_used_only", action="store_false")
    p.set_defaults(filter_used_only=True)

    p.add_argument("--time-filter", action="store_true", help="Enable completion-time QC if a time column is detected")
    p.add_argument("--no-time-filter", dest="time_filter", action="store_false")
    p.set_defaults(time_filter=False)
    p.add_argument("--min-time-sec", type=float, default=60.0, help="Minimum completion time in seconds")
    p.add_argument("--max-time-sec", type=float, default=3600.0, help="Maximum completion time in seconds")

    p.add_argument("--drop-straightliners", action="store_true", help="Drop rows with the same answer across Q2-Q15")
    p.add_argument("--no-drop-straightliners", dest="drop_straightliners", action="store_false")
    p.set_defaults(drop_straightliners=False)
    p.add_argument("--min-answered", type=int, default=8, help="Minimum answered items required to evaluate straight-lining")
    p.add_argument(
        "--drop-invalid-questionnaires",
        action="store_true",
        help="Exclude an entire questionnaire if any Q2-Q15 response is missing, unparseable, or outside 1-7",
    )
    p.add_argument("--no-drop-invalid-questionnaires", dest="drop_invalid_questionnaires", action="store_false")
    p.set_defaults(drop_invalid_questionnaires=True)
    p.add_argument("--expected-human-n", type=int, default=None, help="Assert the final Human analysis sample size")
    p.add_argument("--expected-eps-n", type=int, default=None, help="Assert the final EPS-human analysis sample size")
    p.add_argument("--received-human", type=int, default=None, help="Questionnaires received in the Human arm")
    p.add_argument("--received-eps", type=int, default=None, help="Questionnaires received in the EPS-human arm")
    p.add_argument("--q1-yes-human", type=int, default=None, help="Human respondents reporting exposure to feedback")
    p.add_argument("--q1-yes-eps", type=int, default=None, help="EPS-human respondents reporting exposure to feedback")
    p.add_argument("--group-human", type=str, default="Human", help="Human group label for tables/plots")
    p.add_argument("--group-eps", type=str, default="EPS-human", help="EPS-human group label for tables/plots")

    return p.parse_args()


def read_sheet_arg(sheet_arg: str):
    if sheet_arg.isdigit():
        return int(sheet_arg)
    return sheet_arg


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    human_xlsx = Path(args.human_xlsx)
    eps_xlsx = Path(args.eps_xlsx)
    if not human_xlsx.exists():
        raise FileNotFoundError(f"Human file not found: {human_xlsx}")
    if not eps_xlsx.exists():
        raise FileNotFoundError(f"EPS-human file not found: {eps_xlsx}")

    sheet = read_sheet_arg(args.sheet)

    df_h = pd.read_excel(human_xlsx, sheet_name=sheet)
    df_e = pd.read_excel(eps_xlsx, sheet_name=sheet)

    # Drop empty ID rows if possible.
    id_h = try_find_id_col(df_h)
    id_e = try_find_id_col(df_e)
    if id_h is not None:
        df_h = df_h.dropna(subset=[id_h]).copy()
    if id_e is not None:
        df_e = df_e.dropna(subset=[id_e]).copy()

    supplied_input_h = len(df_h)
    supplied_input_e = len(df_e)

    # Identify question columns.
    q1_h = find_single_col(df_h, Q1_PATTERN)
    q1_e = find_single_col(df_e, Q1_PATTERN)

    q_cols_h = sorted(pick_question_cols(df_h, Q2_15_PATTERN), key=qnum)
    q_cols_e = sorted(pick_question_cols(df_e, Q2_15_PATTERN), key=qnum)

    target = list(range(2, 16))
    if [qnum(c) for c in q_cols_h] != target:
        raise ValueError(f"Human file: expected Q2..Q15 columns, got {[qnum(c) for c in q_cols_h]}")
    if [qnum(c) for c in q_cols_e] != target:
        raise ValueError(f"EPS file: expected Q2..Q15 columns, got {[qnum(c) for c in q_cols_e]}")

    # Q1 screening filter.
    if args.filter_used_only:
        use_h = df_h[q1_h].map(parse_yes_no)
        use_e = df_e[q1_e].map(parse_yes_no)
        df_h = df_h.loc[use_h == 1.0].copy()
        df_e = df_e.loc[use_e == 1.0].copy()

    q1_screened_h = len(df_h)
    q1_screened_e = len(df_e)

    # Completion-time QC (optional, only if detected).
    removed_time_h = removed_time_e = 0
    if args.time_filter:
        tcol_h = try_find_time_col(df_h)
        tcol_e = try_find_time_col(df_e)
        if (tcol_h is not None) and (tcol_e is not None):
            df_h, removed_time_h = apply_time_filter(df_h, tcol_h, args.min_time_sec, args.max_time_sec)
            df_e, removed_time_e = apply_time_filter(df_e, tcol_e, args.min_time_sec, args.max_time_sec)

    # Optional straight-lining QC. The manuscript plotting script did not apply
    # this by default; keep it available for sensitivity checks.
    removed_sl_h = removed_sl_e = 0
    if args.drop_straightliners:
        df_h, removed_sl_h = drop_straightliners(df_h, q_cols_h, min_answered=args.min_answered)
        df_e, removed_sl_e = drop_straightliners(df_e, q_cols_e, min_answered=args.min_answered)

    invalid_audit_h = pd.DataFrame()
    invalid_audit_e = pd.DataFrame()
    if args.drop_invalid_questionnaires:
        df_h, invalid_audit_h = drop_invalid_questionnaires(df_h, q_cols_h, args.group_human)
        df_e, invalid_audit_e = drop_invalid_questionnaires(df_e, q_cols_e, args.group_eps)
    removed_invalid_h = len(invalid_audit_h)
    removed_invalid_e = len(invalid_audit_e)

    if args.expected_human_n is not None and len(df_h) != args.expected_human_n:
        raise AssertionError(f"Expected Human n={args.expected_human_n}, observed n={len(df_h)}")
    if args.expected_eps_n is not None and len(df_e) != args.expected_eps_n:
        raise AssertionError(f"Expected EPS-human n={args.expected_eps_n}, observed n={len(df_e)}")

    # Summaries.
    sum_h = summarize_group(df_h, q_cols_h, args.group_human)
    sum_e = summarize_group(df_e, q_cols_e, args.group_eps)

    # P-values and Holm adjustment.
    pvals = compute_item_pvalues(df_h, df_e, q_cols_h, q_cols_e)
    wide = make_wide_table(sum_h, sum_e, pvals)

    if args.drop_invalid_questionnaires:
        if set(wide["n_human"]) != {len(df_h)}:
            raise AssertionError("Human item-level sample sizes do not match the complete-questionnaire sample")
        if set(wide["n_eps"]) != {len(df_e)}:
            raise AssertionError("EPS-human item-level sample sizes do not match the complete-questionnaire sample")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_wide_csv = outdir / f"{args.prefix}_radar_means_ci_with_p.csv"
    out_long_csv = outdir / f"{args.prefix}_radar_means_ci.csv"
    out_radar_pdf = outdir / f"{args.prefix}_radar_mean_ci.pdf"
    out_radar_png = outdir / f"{args.prefix}_radar_mean_ci.png"
    out_diff_csv = outdir / f"{args.prefix}_item_mean_differences.csv"
    out_diff_pdf = outdir / f"{args.prefix}_item_mean_differences.pdf"
    out_diff_png = outdir / f"{args.prefix}_item_mean_differences.png"
    out_dist_csv = outdir / f"{args.prefix}_domain_response_distribution.csv"
    out_dist_pdf = outdir / f"{args.prefix}_domain_response_distribution.pdf"
    out_dist_png = outdir / f"{args.prefix}_domain_response_distribution.png"
    out_item_results_csv = outdir / f"{args.prefix}_item_level_results.csv"
    out_qc_flow_csv = outdir / f"{args.prefix}_questionnaire_qc_flow.csv"
    out_invalid_audit_csv = outdir / f"{args.prefix}_invalid_questionnaire_audit.csv"

    wide.to_csv(out_wide_csv, index=False, encoding="utf-8-sig")
    pd.concat([sum_h, sum_e], ignore_index=True).to_csv(out_long_csv, index=False, encoding="utf-8-sig")
    diff_df = build_item_difference_table(df_h, df_e, q_cols_h, q_cols_e)
    dist_df = build_domain_distribution_table(
        df_h,
        df_e,
        q_cols_h,
        q_cols_e,
        args.group_human,
        args.group_eps,
    )
    diff_df.to_csv(out_diff_csv, index=False, encoding="utf-8-sig")
    dist_df.to_csv(out_dist_csv, index=False, encoding="utf-8-sig")

    item_results = diff_df.merge(wide, on="question", how="inner", validate="one_to_one").rename(columns={
        "p_value_welch": "p_unadjusted_welch",
        "p_holm_14tests": "p_holm_adjusted_14_items",
    })
    item_results = item_results[[
        "question", "label", "domain",
        "n_human", "mean_human", "ci_low_human", "ci_high_human",
        "n_eps", "mean_eps", "ci_low_eps", "ci_high_eps",
        "mean_difference", "ci_low", "ci_high",
        "p_unadjusted_welch", "p_holm_adjusted_14_items",
    ]]
    item_results.to_csv(out_item_results_csv, index=False, encoding="utf-8-sig")

    invalid_audit = pd.concat([invalid_audit_h, invalid_audit_e], ignore_index=True)
    invalid_audit.to_csv(out_invalid_audit_csv, index=False, encoding="utf-8-sig")

    def prior_qc_exclusions(q1_yes: Optional[int], supplied_rows: int) -> float:
        return np.nan if q1_yes is None else q1_yes - supplied_rows

    qc_flow = pd.DataFrame([
        {
            "group": args.group_human,
            "received": args.received_human,
            "q1_yes": args.q1_yes_human,
            "supplied_precleaned_rows": supplied_input_h,
            "prior_time_or_straightline_exclusions": prior_qc_exclusions(args.q1_yes_human, supplied_input_h),
            "q1_screened_rows_this_run": q1_screened_h,
            "time_exclusions_this_run": removed_time_h,
            "straightline_exclusions_this_run": removed_sl_h,
            "invalid_likert_questionnaires_excluded": removed_invalid_h,
            "final_analysis_n": len(df_h),
        },
        {
            "group": args.group_eps,
            "received": args.received_eps,
            "q1_yes": args.q1_yes_eps,
            "supplied_precleaned_rows": supplied_input_e,
            "prior_time_or_straightline_exclusions": prior_qc_exclusions(args.q1_yes_eps, supplied_input_e),
            "q1_screened_rows_this_run": q1_screened_e,
            "time_exclusions_this_run": removed_time_e,
            "straightline_exclusions_this_run": removed_sl_e,
            "invalid_likert_questionnaires_excluded": removed_invalid_e,
            "final_analysis_n": len(df_e),
        },
    ])
    qc_flow.to_csv(out_qc_flow_csv, index=False, encoding="utf-8-sig")

    # Plot Fig. 3a-c.
    radar_plot_with_ci(
        sum_h=sum_h,
        sum_e=sum_e,
        out_pdf=out_radar_pdf,
        out_png=out_radar_png,
        labels=DEFAULT_LABELS,
        group_h_name=args.group_human,
        group_e_name=args.group_eps,
    )
    plot_item_differences(diff_df, wide["p_holm_14tests"].to_numpy(), out_diff_pdf, out_diff_png)
    plot_domain_distributions(dist_df, out_dist_pdf, out_dist_png, args.group_human, args.group_eps)

    # Console report.
    print("=== Participant-reported outcomes (Phase 2) ===")
    print(f"Human: n={len(df_h)} | removed_time={removed_time_h} | removed_straightline={removed_sl_h} | removed_invalid={removed_invalid_h}")
    print(f"EPS  : n={len(df_e)} | removed_time={removed_time_e} | removed_straightline={removed_sl_e} | removed_invalid={removed_invalid_e}")
    print(f"Saved: {out_wide_csv}")
    print(f"Saved: {out_long_csv}")
    print(f"Saved: {out_radar_pdf}")
    print(f"Saved: {out_diff_pdf}")
    print(f"Saved: {out_dist_pdf}")
    print(f"Saved: {out_diff_csv}")
    print(f"Saved: {out_dist_csv}")
    print(f"Saved: {out_item_results_csv}")
    print(f"Saved: {out_qc_flow_csv}")
    print(f"Saved: {out_invalid_audit_csv}")
    print()
    print(wide[["question", "mean_human", "mean_eps", "p_value_welch", "p_holm_14tests"]].to_string(index=False))


if __name__ == "__main__":
    main()
