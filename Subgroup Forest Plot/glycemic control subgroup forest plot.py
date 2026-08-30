import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


# =========================
# Global plot style
# =========================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
plt.rcParams["mathtext.bfit"] = "Times New Roman:italic:bold"
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
def t_crit_975(df: float) -> float:
    return float(stats.t.ppf(0.975, df))


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


def interaction_pvalue(df: pd.DataFrame, subgroup_col: str) -> float:
    """
    Omnibus group-by-subgroup interaction P value via nested OLS models.
    """
    d = df[["outcome", "group", subgroup_col]].dropna().copy()
    if d.shape[0] < 12:
        raise RuntimeError(f"Insufficient complete observations for interaction test: {subgroup_col}")
    try:
        m0 = smf.ols(f"outcome ~ group + C({subgroup_col})", data=d).fit()
        m1 = smf.ols(f"outcome ~ group * C({subgroup_col})", data=d).fit()
        an = sm.stats.anova_lm(m0, m1)
        return float(an.loc[1, "Pr(>F)"])
    except Exception as exc:
        raise RuntimeError(f"Interaction test failed for subgroup: {subgroup_col}") from exc


def omnibus_interaction_results(df: pd.DataFrame, subgroup_specs: list[tuple[str, str]]) -> pd.DataFrame:
    """Return raw and Holm-adjusted P values for four omnibus interactions."""
    if len(subgroup_specs) != 4:
        raise RuntimeError(f"Expected four subgroup variables, found {len(subgroup_specs)}.")
    raw_p = np.asarray(
        [interaction_pvalue(df, subgroup_col) for _, subgroup_col in subgroup_specs],
        dtype=float,
    )
    if not np.isfinite(raw_p).all():
        raise RuntimeError("All four omnibus interaction P values must be finite.")
    holm_p = multipletests(raw_p, method="holm")[1]
    return pd.DataFrame(
        {
            "Subgroup": [name for name, _ in subgroup_specs],
            "P_interaction_raw": raw_p,
            "P_interaction_holm": holm_p,
        }
    )


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
    return eps_n, hum_n, diff, low, high


def build_subgroup_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []

    eps_n, hum_n, diff, low, high = safe_welch_effect(df)
    records.append(
        {
            "Subgroup": "Overall",
            "Level": "All participants",
            "EPS_n": eps_n,
            "Human_n": hum_n,
            "Effect": diff,
            "CI_low": low,
            "CI_high": high,
        }
    )

    subgroup_specs = [
        ("Sex", "sex_cat"),
        ("Baseline BMI", "bmi_cat"),
        ("Baseline fasting glucose", "fpg0_cat"),
        ("Age", "age_cat"),
    ]

    interaction_tests = omnibus_interaction_results(df, subgroup_specs)

    for sg_name, sg_col in subgroup_specs:
        levels = list(df[sg_col].cat.categories)
        for lv in levels:
            dsub = df[df[sg_col] == lv]
            eps_n, hum_n, diff, low, high = safe_welch_effect(dsub)
            records.append(
                {
                    "Subgroup": sg_name,
                    "Level": str(lv),
                    "EPS_n": eps_n,
                    "Human_n": hum_n,
                    "Effect": diff,
                    "CI_low": low,
                    "CI_high": high,
                }
            )

    return pd.DataFrame.from_records(records), interaction_tests


# =========================
# Forest plot
# =========================
def forest_plot(
    df_subg: pd.DataFrame,
    interaction_tests: pd.DataFrame,
    outfile_png: Path | None,
    outfile_pdf: Path | None,
    title: str | None,
) -> None:
    p_map = interaction_tests.set_index("Subgroup")["P_interaction_holm"]
    subgroup_order = interaction_tests["Subgroup"].tolist()
    observed_subgroups = set(df_subg.loc[df_subg["Subgroup"] != "Overall", "Subgroup"])
    if set(p_map.index) != observed_subgroups:
        raise RuntimeError("Interaction-test rows do not match subgroup variables.")

    overall_rows = df_subg[df_subg["Subgroup"] == "Overall"]
    if len(overall_rows) != 1:
        raise RuntimeError("Expected exactly one overall-effect row.")

    display_rows = []
    overall = overall_rows.iloc[0].to_dict()
    overall.update({"Row_type": "overall", "Label": "Overall", "P_display": np.nan})
    display_rows.append(overall)
    for subgroup_name in subgroup_order:
        display_rows.append(
            {
                "Subgroup": subgroup_name,
                "Level": "",
                "EPS_n": np.nan,
                "Human_n": np.nan,
                "Effect": np.nan,
                "CI_low": np.nan,
                "CI_high": np.nan,
                "Row_type": "header",
                "Label": subgroup_name,
                "P_display": p_map.loc[subgroup_name],
            }
        )
        for _, source_row in df_subg[df_subg["Subgroup"] == subgroup_name].iterrows():
            row = source_row.to_dict()
            row.update({"Row_type": "level", "Label": source_row["Level"], "P_display": np.nan})
            display_rows.append(row)

    d = pd.DataFrame(display_rows)
    y = np.arange(d.shape[0])
    fig_h = max(6.4, 0.39 * d.shape[0] + 1.0)
    fig, ax = plt.subplots(figsize=(12.5, fig_h))

    level_mask = d["Row_type"] == "level"
    overall_mask = d["Row_type"] == "overall"
    mask = level_mask & d["Effect"].notna() & d["CI_low"].notna() & d["CI_high"].notna()
    ax.errorbar(
        d.loc[mask, "Effect"],
        y[mask.to_numpy()],
        xerr=[
            d.loc[mask, "Effect"] - d.loc[mask, "CI_low"],
            d.loc[mask, "CI_high"] - d.loc[mask, "Effect"],
        ],
        fmt="o", color="#2F6B9A", ecolor="#2F6B9A", markersize=4.5,
        capsize=3, linewidth=1,
    )
    overall_row = d.loc[overall_mask].iloc[0]
    ax.errorbar(
        float(overall_row["Effect"]),
        int(y[overall_mask.to_numpy()][0]),
        xerr=[
            [float(overall_row["Effect"] - overall_row["CI_low"])],
            [float(overall_row["CI_high"] - overall_row["Effect"])],
        ],
        fmt="D", color="#202020", ecolor="#202020", markersize=5,
        capsize=3, linewidth=1.1,
    )

    ax.axvline(0, color="#666666", linewidth=0.9)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    ax.set_axisbelow(True)
    for yy in y[d["Row_type"].eq("header").to_numpy()]:
        ax.axhline(yy - 0.5, color="#D9D9D9", linewidth=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(d["Label"], fontsize=9.5)
    for tick, row_type in zip(ax.get_yticklabels(), d["Row_type"]):
        if row_type in {"overall", "header"}:
            tick.set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=7)
    if title:
        ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#202020")
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_xlabel("")

    def fmt_ci(row: pd.Series) -> str:
        if pd.isna(row["Effect"]) or pd.isna(row["CI_low"]) or pd.isna(row["CI_high"]):
            return ""
        return f'{row["Effect"]:.2f} ({row["CI_low"]:.2f}, {row["CI_high"]:.2f})'

    n_x = 1.04
    effect_x = 1.53
    interaction_x = 2.20
    ax.text(
        n_x, 1.02, r"$\mathbfit{n}$ (EPS-human/Human)", transform=ax.transAxes,
        va="bottom", ha="center", fontsize=9, fontweight="bold", clip_on=False,
    )
    ax.text(
        effect_x, 1.02, "Mean difference (95% CI)", transform=ax.transAxes,
        va="bottom", ha="center", fontsize=9, fontweight="bold", clip_on=False,
    )
    ax.text(
        interaction_x, 1.02, r"Interaction $\mathbfit{P}$" "\n(Holm-adjusted)",
        transform=ax.transAxes, va="bottom", ha="center", fontsize=9,
        fontweight="bold", clip_on=False,
    )
    for i, row in d.iterrows():
        yy = y[i]
        if row["Row_type"] in {"overall", "level"}:
            ax.text(
                n_x, yy, f'{int(row["EPS_n"])}/{int(row["Human_n"])}',
                transform=ax.get_yaxis_transform(), va="center", ha="center",
                fontsize=9, clip_on=False,
            )
            ax.text(
                effect_x, yy, fmt_ci(row), transform=ax.get_yaxis_transform(),
                va="center", ha="center", fontsize=9, clip_on=False,
            )
        elif row["Row_type"] == "header":
            ax.text(
                interaction_x, yy, f'{row["P_display"]:.3f}',
                transform=ax.get_yaxis_transform(), va="center", ha="center",
                fontsize=9, clip_on=False,
            )

    ax.invert_yaxis()
    ax.plot(
        [-0.55, 2.48], [0, 0], transform=ax.transAxes, color="#B8B8B8",
        linewidth=0.7, clip_on=False, zorder=0,
    )
    fig.subplots_adjust(left=0.31, right=0.57, top=0.89, bottom=0.14)
    fig.text(
        0.56, 0.025,
        "Mean difference in fasting glucose reduction (mmol/L): EPS-human - Human",
        ha="center", va="bottom", fontsize=10,
    )

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
    parser = argparse.ArgumentParser(description="Glycemic-control subgroup forest plot (Supplementary Fig. 2; fasting glucose reduction).")
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

    subg, interaction_tests = build_subgroup_table(df_all)

    out_table = Path(args.out_table).expanduser()
    out_png = Path(args.out_png).expanduser()
    out_pdf = Path(args.out_pdf).expanduser()

    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_table, engine="openpyxl") as w:
        subg.to_excel(w, sheet_name="Subgroup effects", index=False)
        interaction_tests.to_excel(w, sheet_name="Interaction tests", index=False)

    forest_plot(
        subg,
        interaction_tests,
        outfile_png=out_png,
        outfile_pdf=out_pdf,
        title=args.title,
    )

    print("Saved subgroup table:", str(out_table))
    print("Saved forest plot (PNG):", str(out_png))
    print("Saved forest plot (PDF):", str(out_pdf))
    print("Omnibus interaction P values (raw and Holm-adjusted):")
    print(interaction_tests.to_string(index=False))


if __name__ == "__main__":
    main()
