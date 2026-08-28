import argparse
import math
from pathlib import Path

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
# Cleaning helpers
# =========================
def clean_numeric(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(series, errors="coerce")


def clean_sex_binary_keepna(series: pd.Series) -> pd.Series:
    """
    Map sex to binary with NaN preserved:
      Female -> 1.0
      Male   -> 0.0
      Unknown/unparseable -> NaN
    """
    if series is None:
        return pd.Series(dtype="float64")

    if pd.api.types.is_numeric_dtype(series):
        s = series.astype("float64")
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        out[s == 0] = 1.0
        out[s == 2] = 1.0
        out[s == 1] = 0.0
        return out

    s = series.astype(str).str.strip().str.lower()

    female_set = {"f", "female", "0", "2"}
    male_set = {"m", "male", "1"}

    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out[s.isin({x.lower() for x in female_set})] = 1.0
    out[s.isin({x.lower() for x in male_set})] = 0.0
    return out


# =========================
# Stats helpers
# =========================
def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def z_crit_975() -> float:
    return 1.959963984540054


def t_crit_975(df: float) -> float:
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except Exception:
        return z_crit_975()


def welch_ci_mean_diff(x_eps, x_hum):
    """
    Mean difference (EPS-human minus Human) with Welch 95% CI.
    Returns: (diff, ci_low, ci_high, df)
    """
    x = pd.Series(x_eps).dropna().astype(float).values
    y = pd.Series(x_hum).dropna().astype(float).values
    n1, n2 = len(x), len(y)

    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, np.nan, np.nan

    m1, m2 = float(np.mean(x)), float(np.mean(y))
    s1, s2 = float(np.std(x, ddof=1)), float(np.std(y, ddof=1))
    diff = m1 - m2

    se2 = (s1 ** 2) / n1 + (s2 ** 2) / n2
    if not np.isfinite(se2) or se2 <= 0:
        return diff, np.nan, np.nan, np.nan

    se = math.sqrt(se2)

    num = se2 ** 2
    den = ((s1 ** 2 / n1) ** 2) / (n1 - 1) + ((s2 ** 2 / n2) ** 2) / (n2 - 1)
    df = num / den if (np.isfinite(den) and den > 0) else float(n1 + n2 - 2)

    tcrit = t_crit_975(df)
    return diff, diff - tcrit * se, diff + tcrit * se, df


def welch_ttest_pvalue(x, y) -> float:
    x = pd.Series(x).dropna().astype(float).values
    y = pd.Series(y).dropna().astype(float).values
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
        if denom <= 0 or not np.isfinite(denom):
            return np.nan
        z = (mx - my) / denom
        return float(2.0 * (1.0 - normal_cdf(abs(z))))


def interaction_pvalue(df: pd.DataFrame, subgroup_col: str) -> float:
    """
    Group × subgroup interaction p-value via nested OLS.
    Returns NaN if statsmodels unavailable or sample too small.
    """
    try:
        import statsmodels.formula.api as smf
        import statsmodels.api as sm
    except Exception:
        return np.nan

    d = df[["outcome", "group", subgroup_col]].dropna().copy()
    if d.shape[0] < 12:
        return np.nan

    try:
        m0 = smf.ols(f"outcome ~ group + C({subgroup_col})", data=d).fit()
        m1 = smf.ols(f"outcome ~ group * C({subgroup_col})", data=d).fit()
        an = sm.stats.anova_lm(m0, m1)
        return float(an.loc[1, "Pr(>F)"])
    except Exception:
        return np.nan


# =========================
# Subgroup bins (match Appendix F)
# =========================
def add_subgroup_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["sex_cat"] = out["sex"].map({1.0: "Female", 0.0: "Male"})
    out["sex_cat"] = pd.Categorical(out["sex_cat"], categories=["Female", "Male"], ordered=True)

    out["bmi_cat"] = pd.cut(
        out["bmi"],
        bins=[-np.inf, 24, np.inf],
        right=False,
        labels=["<24", "≥24"],
    )
    out["bmi_cat"] = pd.Categorical(
        out["bmi_cat"],
        categories=["<24", "≥24"],
        ordered=True,
    )

    fpg = out["fpg0"].astype(float)
    labels_fpg = ["<5.9 mmol/L", "5.9–6.6 mmol/L", "≥6.6 mmol/L"]
    out["fpg0_cat"] = pd.cut(
        fpg,
        bins=[-np.inf, 5.9, 6.6, np.inf],
        right=False,
        labels=labels_fpg,
    )
    out["fpg0_cat"] = pd.Categorical(out["fpg0_cat"], categories=labels_fpg, ordered=True)

    out["age_cat"] = pd.cut(
        out["age"],
        bins=[-np.inf, 45, np.inf],
        right=False,
        labels=["<45", "≥45"],
    )
    out["age_cat"] = pd.Categorical(out["age_cat"], categories=["<45", "≥45"], ordered=True)

    return out


# =========================
# Subgroup effects table
# =========================
def safe_welch_effect(df_sub: pd.DataFrame):
    eps = df_sub.loc[df_sub["group"] == 1, "outcome"]
    hum = df_sub.loc[df_sub["group"] == 0, "outcome"]
    eps_n = int(eps.dropna().shape[0])
    hum_n = int(hum.dropna().shape[0])
    diff, low, high, _ = welch_ci_mean_diff(eps, hum)
    p = welch_ttest_pvalue(eps, hum)
    return eps_n, hum_n, diff, low, high, p


def build_subgroup_table(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    eps_n, hum_n, diff, low, high, p = safe_welch_effect(df)
    records.append(
        {
            "Subgroup": "Overall",
            "Level": "All participants",
            "EPS_n": eps_n,
            "Human_n": hum_n,
            "Effect": diff,
            "CI_low": low,
            "CI_high": high,
            "P_value": p,
            "P_interaction": np.nan,
        }
    )

    subgroup_specs = [
        ("Sex", "sex_cat"),
        ("Baseline BMI", "bmi_cat"),
        ("Baseline fasting glucose", "fpg0_cat"),
        ("Age", "age_cat"),
    ]

    for sg_name, sg_col in subgroup_specs:
        p_int = interaction_pvalue(df, sg_col)
        levels = list(df[sg_col].cat.categories)
        for lv in levels:
            dsub = df[df[sg_col] == lv]
            eps_n, hum_n, diff, low, high, p = safe_welch_effect(dsub)
            records.append(
                {
                    "Subgroup": sg_name,
                    "Level": str(lv),
                    "EPS_n": eps_n,
                    "Human_n": hum_n,
                    "Effect": diff,
                    "CI_low": low,
                    "CI_high": high,
                    "P_value": p,
                    "P_interaction": p_int,
                }
            )

    out = pd.DataFrame.from_records(records)

    def fmt_ci_row(r) -> str:
        if pd.isna(r["Effect"]) or pd.isna(r["CI_low"]) or pd.isna(r["CI_high"]):
            return ""
        return f'{r["Effect"]:.2f} ({r["CI_low"]:.2f}, {r["CI_high"]:.2f})'

    out["Effect_CI"] = out.apply(fmt_ci_row, axis=1)
    out["N_text"] = out.apply(lambda r: f'N={int(r["EPS_n"])}/{int(r["Human_n"])}', axis=1)
    return out


# =========================
# Forest plot
# =========================
def forest_plot(df_subg: pd.DataFrame, outfile_png: Path | None, outfile_pdf: Path | None, title: str | None):
    d = df_subg.copy()

    d["Label"] = np.where(d["Subgroup"] == "Overall", "Overall", d["Subgroup"] + ": " + d["Level"])

    n_rows = d.shape[0]
    y = np.arange(n_rows)

    fig_h = max(6.0, 0.38 * n_rows + 1.2)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    mask = d["Effect"].notna() & d["CI_low"].notna() & d["CI_high"].notna()
    y_mask = y[mask.to_numpy()]

    if y_mask.size > 0:
        ax.errorbar(
            d.loc[mask, "Effect"].to_numpy(),
            y_mask,
            xerr=[
                (d.loc[mask, "Effect"] - d.loc[mask, "CI_low"]).to_numpy(),
                (d.loc[mask, "CI_high"] - d.loc[mask, "Effect"]).to_numpy(),
            ],
            fmt="o",
            capsize=3,
            linewidth=1,
        )

    ax.axvline(0, linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(d["Label"])
    ax.set_xlabel("Mean difference in fasting glucose reduction (mmol/L): EPS-human − Human")
    if title:
        ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    xmin_candidates = [0.0]
    xmax_candidates = [0.0]
    if d["CI_low"].notna().any():
        xmin_candidates.append(float(np.nanmin(d["CI_low"].to_numpy())))
    if d["CI_high"].notna().any():
        xmax_candidates.append(float(np.nanmax(d["CI_high"].to_numpy())))

    xmin = min(xmin_candidates)
    xmax = max(xmax_candidates)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
        xmin, xmax = -1.0, 1.0

    base_span = xmax - xmin
    pad = 0.12 * base_span
    xmin_plot = xmin - pad
    xmax_plot = xmax + pad

    text_space = 0.60 * (xmax_plot - xmin_plot)
    ax.set_xlim(xmin_plot, xmax_plot + text_space)

    def fmt_ci(r) -> str:
        if pd.isna(r["Effect"]) or pd.isna(r["CI_low"]) or pd.isna(r["CI_high"]):
            return ""
        return f'{r["Effect"]:.2f} ({r["CI_low"]:.2f}, {r["CI_high"]:.2f})'

    x_text = ax.get_xlim()[1]
    for i, r in d.iterrows():
        ax.text(
            x_text,
            y[i],
            f'{r["N_text"]}   {fmt_ci(r)}',
            va="center",
            ha="right",
            fontsize=9,
        )

    ax.invert_yaxis()
    plt.tight_layout()

    if outfile_png is not None:
        plt.savefig(outfile_png, dpi=300)
    if outfile_pdf is not None:
        plt.savefig(outfile_pdf)
    plt.close(fig)


# =========================
# IO helpers
# =========================
def require_columns(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def read_excel_clean(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()
    return df


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Glycemic-control subgroup forest plot (Extended Data Fig. 2; fasting glucose reduction).")
    parser.add_argument("--eps", required=True, type=str, help="Path to EPS-human arm Excel file.")
    parser.add_argument("--human", required=True, type=str, help="Path to Human arm Excel file.")
    parser.add_argument("--out_table", required=True, type=str, help="Output Excel path for subgroup table.")
    parser.add_argument("--out_png", required=True, type=str, help="Output PNG path for forest plot.")
    parser.add_argument("--out_pdf", required=True, type=str, help="Output PDF path for forest plot.")
    parser.add_argument("--col_age", default="age", type=str)
    parser.add_argument("--col_sex", default="sex", type=str)
    parser.add_argument("--col_bmi", default="bmi", type=str)
    parser.add_argument("--col_fpg0", default="baseline_fpg_mmol", type=str)
    parser.add_argument("--col_fpg1", default="endpoint_fpg_mmol", type=str)
    parser.add_argument("--title", default=None, type=str)
    args = parser.parse_args()

    file_eps = Path(args.eps).expanduser()
    file_hum = Path(args.human).expanduser()

    df_eps = read_excel_clean(file_eps)
    df_hum = read_excel_clean(file_hum)

    required = [args.col_age, args.col_sex, args.col_bmi, args.col_fpg0, args.col_fpg1]
    require_columns(df_eps, required, "EPS-human file")
    require_columns(df_hum, required, "Human file")

    eps_age = clean_numeric(df_eps[args.col_age])
    hum_age = clean_numeric(df_hum[args.col_age])

    eps_bmi = clean_numeric(df_eps[args.col_bmi])
    hum_bmi = clean_numeric(df_hum[args.col_bmi])

    eps_sex = clean_sex_binary_keepna(df_eps[args.col_sex])
    hum_sex = clean_sex_binary_keepna(df_hum[args.col_sex])

    eps_fpg0 = clean_numeric(df_eps[args.col_fpg0])
    hum_fpg0 = clean_numeric(df_hum[args.col_fpg0])

    eps_fpg1 = clean_numeric(df_eps[args.col_fpg1])
    hum_fpg1 = clean_numeric(df_hum[args.col_fpg1])

    eps_outcome = eps_fpg0 - eps_fpg1
    hum_outcome = hum_fpg0 - hum_fpg1

    eps_df = pd.DataFrame(
        {
            "group": 1,
            "outcome": eps_outcome,
            "age": eps_age,
            "bmi": eps_bmi,
            "sex": eps_sex,
            "fpg0": eps_fpg0,
        }
    )
    hum_df = pd.DataFrame(
        {
            "group": 0,
            "outcome": hum_outcome,
            "age": hum_age,
            "bmi": hum_bmi,
            "sex": hum_sex,
            "fpg0": hum_fpg0,
        }
    )

    df_all = pd.concat([eps_df, hum_df], ignore_index=True)
    df_all = add_subgroup_columns(df_all)

    subg = build_subgroup_table(df_all)

    out_table = Path(args.out_table).expanduser()
    out_png = Path(args.out_png).expanduser()
    out_pdf = Path(args.out_pdf).expanduser()

    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_table, engine="openpyxl") as w:
        subg.to_excel(w, sheet_name="Subgroup effects", index=False)

    forest_plot(subg, outfile_png=out_png, outfile_pdf=out_pdf, title=args.title)

    print("Saved subgroup table:", str(out_table))
    print("Saved forest plot (PNG):", str(out_png))
    print("Saved forest plot (PDF):", str(out_pdf))


if __name__ == "__main__":
    main()
