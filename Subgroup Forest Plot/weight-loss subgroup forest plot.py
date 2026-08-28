import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Global plot style
# =========================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# =========================
# Column config (dataset-specific)
# =========================
COL_OUTCOME = "weight_loss_kg"   # primary outcome (kg): weight change
COL_AGE = "age"
COL_SEX = "sex"
COL_BMI = "bmi"

BASELINE_WEIGHT_CANDIDATES = [
    "baseline_weight_kg",
    "baseline_weight",
    "weight_baseline",
]


# =========================
# Utilities
# =========================
def pick_first_existing(columns: List[str], candidates: List[str]) -> Optional[str]:
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clean_sex_binary(series: pd.Series) -> pd.Series:
    """
    Map sex to binary: Female=1, Male=0. Unknown stays NaN.
    Accepts common Chinese/English strings and numeric codes.
    """
    s = series.copy()

    # Preserve missing values
    out = pd.Series(np.nan, index=s.index, dtype=float)

    # Normalize
    raw = s.astype(str).str.strip().str.lower()

    female_tokens = {"female", "f"}
    male_tokens = {"male", "m"}

    # Numeric codes seen in some exports:
    # Keep the user's original convention: 0/2 as female, 1 as male.
    # If your data uses another convention, override via preprocessing or adapt here.
    female_codes = {"0", "2"}
    male_codes = {"1"}

    out[raw.isin(female_tokens | female_codes)] = 1.0
    out[raw.isin(male_tokens | male_codes)] = 0.0
    return out


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def z_crit_975() -> float:
    return 1.959963984540054


def t_crit_975(df: float) -> float:
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except Exception:
        return z_crit_975()


def welch_ci_mean_diff(x_eps: pd.Series, x_hum: pd.Series) -> Tuple[float, float, float, float]:
    """
    Mean difference (EPS - Human) with Welch CI.
    Returns: diff, ci_low, ci_high, df
    """
    x = pd.Series(x_eps).dropna().astype(float).values
    y = pd.Series(x_hum).dropna().astype(float).values
    n1, n2 = len(x), len(y)

    if n1 < 2 or n2 < 2:
        return (np.nan, np.nan, np.nan, np.nan)

    m1, m2 = float(np.mean(x)), float(np.mean(y))
    s1, s2 = float(np.std(x, ddof=1)), float(np.std(y, ddof=1))
    diff = m1 - m2

    se2 = (s1**2) / n1 + (s2**2) / n2
    if se2 <= 0:
        return (diff, np.nan, np.nan, np.nan)

    se = math.sqrt(se2)

    num = se2**2
    den = ((s1**2 / n1) ** 2) / (n1 - 1) + ((s2**2 / n2) ** 2) / (n2 - 1)
    df = num / den if den > 0 else float(n1 + n2 - 2)

    tcrit = t_crit_975(df)
    ci_low = diff - tcrit * se
    ci_high = diff + tcrit * se
    return (float(diff), float(ci_low), float(ci_high), float(df))


def welch_ttest_pvalue(x_eps: pd.Series, x_hum: pd.Series) -> float:
    x = pd.Series(x_eps).dropna().astype(float).values
    y = pd.Series(x_hum).dropna().astype(float).values
    if len(x) < 2 or len(y) < 2:
        return np.nan

    try:
        from scipy import stats
        _, p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
        return float(p)
    except Exception:
        mx, my = float(np.mean(x)), float(np.mean(y))
        vx, vy = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
        denom = math.sqrt(vx / len(x) + vy / len(y))
        if denom == 0:
            return np.nan
        z = (mx - my) / denom
        p = 2.0 * (1.0 - normal_cdf(abs(z)))
        return float(p)


def interaction_pvalue(df: pd.DataFrame, subgroup_col: str) -> float:
    """
    Group × subgroup interaction p-value via nested OLS models.
    Returns NaN if statsmodels is not available or data are insufficient.
    """
    try:
        import statsmodels.formula.api as smf
        import statsmodels.api as sm
    except Exception:
        return np.nan

    d = df[["outcome", "group", subgroup_col]].dropna().copy()
    if d.shape[0] < 30:
        return np.nan

    try:
        m0 = smf.ols(f"outcome ~ group + C({subgroup_col})", data=d).fit()
        m1 = smf.ols(f"outcome ~ group * C({subgroup_col})", data=d).fit()
        an = sm.stats.anova_lm(m0, m1)
        return float(an.loc[1, "Pr(>F)"])
    except Exception:
        return np.nan


# =========================
# Data preparation
# =========================
def load_arm(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip().str.lower()
    return df


def build_analysis_df(df_eps: pd.DataFrame, df_hum: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    # Baseline weight column
    col_w0_eps = pick_first_existing(list(df_eps.columns), BASELINE_WEIGHT_CANDIDATES)
    col_w0_hum = pick_first_existing(list(df_hum.columns), BASELINE_WEIGHT_CANDIDATES)
    if col_w0_eps is None or col_w0_hum is None:
        raise ValueError(
            "Baseline weight column not found in both files. "
            f"Candidates={BASELINE_WEIGHT_CANDIDATES}. "
            f"EPS_found={col_w0_eps}, Human_found={col_w0_hum}"
        )

    # Required columns
    required = [COL_OUTCOME, COL_AGE, COL_SEX, COL_BMI]
    for c in required:
        if c not in df_eps.columns:
            raise ValueError(f"Missing column in EPS file: {c}")
        if c not in df_hum.columns:
            raise ValueError(f"Missing column in Human file: {c}")

    eps_df = pd.DataFrame(
        {
            "group": 1,  # EPS-human
            "outcome": clean_numeric(df_eps[COL_OUTCOME]),
            "age": clean_numeric(df_eps[COL_AGE]),
            "bmi": clean_numeric(df_eps[COL_BMI]),
            "sex": clean_sex_binary(df_eps[COL_SEX]),
            "w0": clean_numeric(df_eps[col_w0_eps]),
        }
    )
    hum_df = pd.DataFrame(
        {
            "group": 0,  # Human
            "outcome": clean_numeric(df_hum[COL_OUTCOME]),
            "age": clean_numeric(df_hum[COL_AGE]),
            "bmi": clean_numeric(df_hum[COL_BMI]),
            "sex": clean_sex_binary(df_hum[COL_SEX]),
            "w0": clean_numeric(df_hum[col_w0_hum]),
        }
    )

    df_all = pd.concat([eps_df, hum_df], ignore_index=True)

    # Subgroup bins
    meta: Dict[str, float] = {}
    df_all, meta_bins = add_subgroup_columns(df_all)
    meta.update(meta_bins)
    return df_all, meta


def add_subgroup_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    out = df.copy()
    meta: Dict[str, float] = {}

    out["sex_cat"] = out["sex"].map({1.0: "Female", 0.0: "Male"})
    out["sex_cat"] = pd.Categorical(out["sex_cat"], categories=["Female", "Male"], ordered=True)

    out["bmi_cat"] = pd.cut(
        out["bmi"],
        bins=[-np.inf, 24, 28, np.inf],
        right=False,
        labels=["<24", "24–27.9", "≥28"],
    )
    out["bmi_cat"] = pd.Categorical(out["bmi_cat"], categories=["<24", "24–27.9", "≥28"], ordered=True)

    out["age_cat"] = pd.cut(
        out["age"],
        bins=[-np.inf, 45, np.inf],
        right=False,
        labels=["<45", "≥45"],
    )
    out["age_cat"] = pd.Categorical(
        out["age_cat"],
        categories=["<45", "≥45"],
        ordered=True,
    )

    # Baseline weight categories using pooled sample terciles
    w0 = out["w0"].astype(float)
    w0_nonmiss = w0.dropna()
    if w0_nonmiss.shape[0] >= 10:
        c1 = float(w0_nonmiss.quantile(1.0 / 3.0))
        c2 = float(w0_nonmiss.quantile(2.0 / 3.0))

        c1_disp = int(round(c1))
        c2_disp = int(round(c2))

        if c2_disp <= c1_disp:
            c2_disp = c1_disp + 1

        meta["w0_tercile_q33"] = c1
        meta["w0_tercile_q67"] = c2
        meta["w0_cut1_used"] = float(c1_disp)
        meta["w0_cut2_used"] = float(c2_disp)

        labels = [
            f"<{c1_disp:g} kg",
            f"{c1_disp:g}–{c2_disp:g} kg",
            f"≥{c2_disp:g} kg",
        ]

        out["w0_cat"] = pd.cut(
            w0,
            bins=[-np.inf, c1_disp, c2_disp, np.inf],
            right=False,
            labels=labels,
        )
        out["w0_cat"] = pd.Categorical(out["w0_cat"], categories=labels, ordered=True)
    else:
        out["w0_cat"] = pd.Categorical([np.nan] * out.shape[0])

    return out, meta


# =========================
# Subgroup effects table
# =========================
def subgroup_effect(df_sub: pd.DataFrame) -> Tuple[int, int, float, float, float, float]:
    eps = df_sub.loc[df_sub["group"] == 1, "outcome"]
    hum = df_sub.loc[df_sub["group"] == 0, "outcome"]
    eps_n = int(eps.dropna().shape[0])
    hum_n = int(hum.dropna().shape[0])

    diff, low, high, _ = welch_ci_mean_diff(eps, hum)
    p = welch_ttest_pvalue(eps, hum)
    return eps_n, hum_n, diff, low, high, p


def build_subgroup_table(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []

    # Row order matches Appendix-style presentation: overall first, then subgroup levels
    records.append(_make_record(df, "Overall", "All participants", p_int=np.nan))

    subgroup_specs = [
        ("Sex", "sex_cat"),
        ("Baseline BMI", "bmi_cat"),
        ("Baseline weight", "w0_cat"),
        ("Age", "age_cat"),
    ]

    for sg_name, sg_col in subgroup_specs:
        if sg_col not in df.columns:
            continue

        # Compute interaction p-value once per subgroup
        p_int = interaction_pvalue(df, sg_col)

        # Skip if the subgroup column has no categories/values
        if df[sg_col].dropna().shape[0] == 0:
            continue

        # Use declared category order if categorical; otherwise use sorted unique values
        if isinstance(df[sg_col].dtype, pd.CategoricalDtype):
            levels = list(df[sg_col].cat.categories)
        else:
            levels = sorted(df[sg_col].dropna().unique().tolist())

        for lv in levels:
            dsub = df[df[sg_col] == lv]
            records.append(_make_record(dsub, sg_name, str(lv), p_int=p_int))

    return pd.DataFrame.from_records(records)


def _make_record(df_sub: pd.DataFrame, subgroup: str, level: str, p_int: float) -> Dict[str, object]:
    eps_n, hum_n, diff, low, high, p = subgroup_effect(df_sub)
    return {
        "Subgroup": subgroup,
        "Level": level,
        "EPS_n": eps_n,
        "Human_n": hum_n,
        "Effect": diff,
        "CI_low": low,
        "CI_high": high,
        "P_value": p,
        "P_interaction": p_int,
    }


def add_forest_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["N_EPS/Human"] = out.apply(lambda r: f'{int(r["EPS_n"])}/{int(r["Human_n"])}', axis=1)

    def fmt_ci(r: pd.Series) -> str:
        if pd.isna(r["Effect"]) or pd.isna(r["CI_low"]) or pd.isna(r["CI_high"]):
            return ""
        return f'{r["Effect"]:.2f} ({r["CI_low"]:.2f}, {r["CI_high"]:.2f})'

    out["Effect (95% CI)"] = out.apply(fmt_ci, axis=1)

    def fmt_label(r: pd.Series) -> str:
        if r["Subgroup"] == "Overall":
            return "Overall"
        return f'{r["Subgroup"]}: {r["Level"]}'

    out["Label"] = out.apply(fmt_label, axis=1)
    return out


# =========================
# Forest plot
# =========================
def forest_plot(
    df_subg: pd.DataFrame,
    outfile_png: Optional[Path],
    outfile_pdf: Optional[Path],
    title: Optional[str] = None,
) -> None:
    d = add_forest_strings(df_subg)

    y = np.arange(d.shape[0])

    fig_h = max(6.0, 0.35 * d.shape[0] + 1.2)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    mask = d["Effect"].notna() & d["CI_low"].notna() & d["CI_high"].notna()

    ax.errorbar(
        d.loc[mask, "Effect"],
        y[mask.values],
        xerr=[
            d.loc[mask, "Effect"] - d.loc[mask, "CI_low"],
            d.loc[mask, "CI_high"] - d.loc[mask, "Effect"],
        ],
        fmt="o",
        capsize=3,
        linewidth=1,
        markeredgewidth=0,
    )

    ax.axvline(0.0, linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(d["Label"])
    ax.set_xlabel("Mean difference in weight change (kg): EPS-human − Human")

    if title:
        ax.set_title(title)

    # Remove outer box
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Put overall at the top
    ax.invert_yaxis()

    # Right-side annotation using axis coordinates
    trans = ax.get_yaxis_transform()  # x in axes coords, y in data coords
    for i, r in d.iterrows():
        yy = y[i]
        right_text = f'N={r["N_EPS/Human"]}   {r["Effect (95% CI)"]}'.rstrip()
        ax.text(
            1.02,
            yy,
            right_text,
            transform=trans,
            va="center",
            ha="left",
            fontsize=9,
            clip_on=False,
        )

    plt.tight_layout()

    if outfile_png is not None:
        ensure_parent_dir(outfile_png)
        plt.savefig(outfile_png, dpi=300, bbox_inches="tight", pad_inches=0.1)

    if outfile_pdf is not None:
        ensure_parent_dir(outfile_pdf)
        plt.savefig(outfile_pdf, bbox_inches="tight", pad_inches=0.1)

    plt.close(fig)


# =========================
# Export
# =========================
def export_tables(
    df_subg: pd.DataFrame,
    meta: Dict[str, float],
    out_xlsx: Path,
) -> None:
    out = df_subg.copy()
    forest = add_forest_strings(df_subg)[
        ["Label", "EPS_n", "Human_n", "N_EPS/Human", "Effect", "CI_low", "CI_high", "Effect (95% CI)", "P_value", "P_interaction"]
    ]

    meta_df = pd.DataFrame(
        [{"Key": k, "Value": v} for k, v in meta.items()]
    ).sort_values("Key")

    ensure_parent_dir(out_xlsx)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        out.to_excel(w, sheet_name="Subgroup effects", index=False)
        forest.to_excel(w, sheet_name="Forest table", index=False)
        meta_df.to_excel(w, sheet_name="Meta", index=False)


# =========================
# Main
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weight-loss subgroup forest plot (Extended Data Fig. 1; EPS-human vs Human).")
    p.add_argument("--eps", type=str, default="data/weight-loss/EPS-Human weight-loss.xlsx", help="EPS-human arm Excel file.")
    p.add_argument("--human", type=str, default="data/weight-loss/Human weight-loss.xlsx", help="Human arm Excel file.")
    p.add_argument("--out-prefix", type=str, default="outputs/weightloss/EPS_vs_Human_weightloss", help="Output prefix path (no extension).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    eps_path = Path(args.eps).expanduser().resolve()
    hum_path = Path(args.human).expanduser().resolve()
    out_prefix = Path(args.out_prefix).expanduser()

    if not eps_path.exists():
        raise FileNotFoundError(f"EPS file not found: {eps_path}")
    if not hum_path.exists():
        raise FileNotFoundError(f"Human file not found: {hum_path}")

    df_eps = load_arm(eps_path)
    df_hum = load_arm(hum_path)

    df_all, meta = build_analysis_df(df_eps, df_hum)
    subg = build_subgroup_table(df_all)

    out_xlsx = out_prefix.with_suffix(".xlsx")
    out_png = out_prefix.with_suffix(".png")
    out_pdf = out_prefix.with_suffix(".pdf")

    export_tables(subg, meta, out_xlsx)
    forest_plot(subg, outfile_png=out_png, outfile_pdf=out_pdf, title=None)

    print("Saved subgroup workbook:", str(out_xlsx))
    print("Saved forest plot (PNG):", str(out_png))
    print("Saved forest plot (PDF):", str(out_pdf))


if __name__ == "__main__":
    main()
