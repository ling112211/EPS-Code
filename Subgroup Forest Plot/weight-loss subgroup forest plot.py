import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def t_crit_975(df: float) -> float:
    return float(stats.t.ppf(0.975, df))


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


def interaction_pvalue(df: pd.DataFrame, subgroup_col: str) -> float:
    """
    Omnibus group-by-subgroup interaction P value via nested OLS models.
    """
    d = df[["outcome", "group", subgroup_col]].dropna().copy()
    if d.shape[0] < 30:
        raise RuntimeError(f"Insufficient complete observations for interaction test: {subgroup_col}")
    try:
        m0 = smf.ols(f"outcome ~ group + C({subgroup_col})", data=d).fit()
        m1 = smf.ols(f"outcome ~ group * C({subgroup_col})", data=d).fit()
        an = sm.stats.anova_lm(m0, m1)
        return float(an.loc[1, "Pr(>F)"])
    except Exception as exc:
        raise RuntimeError(f"Interaction test failed for subgroup: {subgroup_col}") from exc


def omnibus_interaction_results(
    df: pd.DataFrame,
    subgroup_specs: List[Tuple[str, str]],
) -> pd.DataFrame:
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
def subgroup_effect(df_sub: pd.DataFrame) -> Tuple[int, int, float, float, float]:
    eps = df_sub.loc[df_sub["group"] == 1, "outcome"]
    hum = df_sub.loc[df_sub["group"] == 0, "outcome"]
    eps_n = int(eps.dropna().shape[0])
    hum_n = int(hum.dropna().shape[0])

    diff, low, high, _ = welch_ci_mean_diff(eps, hum)
    return eps_n, hum_n, diff, low, high


def build_subgroup_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records: List[Dict[str, object]] = []
    records.append(_make_record(df, "Overall", "All participants"))

    subgroup_specs = [
        ("Sex", "sex_cat"),
        ("Baseline BMI", "bmi_cat"),
        ("Baseline weight", "w0_cat"),
        ("Age", "age_cat"),
    ]

    interaction_tests = omnibus_interaction_results(df, subgroup_specs)

    for sg_name, sg_col in subgroup_specs:
        if df[sg_col].dropna().shape[0] == 0:
            raise RuntimeError(f"Subgroup variable contains no valid values: {sg_col}")
        levels = list(df[sg_col].cat.categories)
        for lv in levels:
            dsub = df[df[sg_col] == lv]
            records.append(_make_record(dsub, sg_name, str(lv)))

    return pd.DataFrame.from_records(records), interaction_tests


def _make_record(df_sub: pd.DataFrame, subgroup: str, level: str) -> Dict[str, object]:
    eps_n, hum_n, diff, low, high = subgroup_effect(df_sub)
    return {
        "Subgroup": subgroup,
        "Level": level,
        "EPS_n": eps_n,
        "Human_n": hum_n,
        "Effect": diff,
        "CI_low": low,
        "CI_high": high,
    }


# =========================
# Forest plot
# =========================
def forest_plot(
    df_subg: pd.DataFrame,
    interaction_tests: pd.DataFrame,
    outfile_png: Optional[Path],
    outfile_pdf: Optional[Path],
    title: Optional[str] = None,
) -> None:
    p_map = interaction_tests.set_index("Subgroup")["P_interaction_holm"]
    subgroup_order = interaction_tests["Subgroup"].tolist()
    observed_subgroups = set(df_subg.loc[df_subg["Subgroup"] != "Overall", "Subgroup"])
    if set(p_map.index) != observed_subgroups:
        raise RuntimeError("Interaction-test rows do not match subgroup variables.")

    overall_rows = df_subg[df_subg["Subgroup"] == "Overall"]
    if len(overall_rows) != 1:
        raise RuntimeError("Expected exactly one overall-effect row.")

    display_rows: List[Dict[str, object]] = []
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
        fmt="o",
        color="#2F6B9A",
        ecolor="#2F6B9A",
        markersize=4.5,
        capsize=3,
        linewidth=1,
    )
    overall_row = d.loc[overall_mask].iloc[0]
    ax.errorbar(
        float(overall_row["Effect"]),
        int(y[overall_mask.to_numpy()][0]),
        xerr=[
            [float(overall_row["Effect"] - overall_row["CI_low"])],
            [float(overall_row["CI_high"] - overall_row["Effect"])],
        ],
        fmt="D",
        color="#202020",
        ecolor="#202020",
        markersize=5,
        capsize=3,
        linewidth=1.1,
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
        0.56, 0.025, "Mean difference in weight change (kg): EPS-human - Human",
        ha="center", va="bottom", fontsize=10,
    )

    if outfile_png is not None:
        ensure_parent_dir(outfile_png)
        plt.savefig(outfile_png, dpi=300)

    if outfile_pdf is not None:
        ensure_parent_dir(outfile_pdf)
        plt.savefig(outfile_pdf)

    plt.close(fig)


# =========================
# Export
# =========================
def export_tables(
    df_subg: pd.DataFrame,
    interaction_tests: pd.DataFrame,
    meta: Dict[str, float],
    out_xlsx: Path,
) -> None:
    out = df_subg.copy()
    meta_df = pd.DataFrame(
        [{"Key": k, "Value": v} for k, v in meta.items()]
    ).sort_values("Key")

    ensure_parent_dir(out_xlsx)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        out.to_excel(w, sheet_name="Subgroup effects", index=False)
        interaction_tests.to_excel(w, sheet_name="Interaction tests", index=False)
        meta_df.to_excel(w, sheet_name="Meta", index=False)


# =========================
# Main
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weight-loss subgroup forest plot (Supplementary Fig. 1; EPS-human vs Human).")
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
    subg, interaction_tests = build_subgroup_table(df_all)

    out_xlsx = out_prefix.with_suffix(".xlsx")
    out_png = out_prefix.with_suffix(".png")
    out_pdf = out_prefix.with_suffix(".pdf")

    export_tables(subg, interaction_tests, meta, out_xlsx)
    forest_plot(
        subg,
        interaction_tests,
        outfile_png=out_png,
        outfile_pdf=out_pdf,
        title=None,
    )

    print("Saved subgroup workbook:", str(out_xlsx))
    print("Saved forest plot (PNG):", str(out_png))
    print("Saved forest plot (PDF):", str(out_pdf))
    print("Omnibus interaction P values (raw and Holm-adjusted):")
    print(interaction_tests.to_string(index=False))


if __name__ == "__main__":
    main()
