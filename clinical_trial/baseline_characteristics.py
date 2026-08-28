from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


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


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clean_sex_binary_keepna(series: pd.Series) -> pd.Series:
    """
    Map sex to binary:
      female = 1, male = 0
    Unrecognized entries become NaN.
    """
    s = series.astype(str).str.strip().str.lower()
    female_set = {"f", "female", "0", "2"}
    male_set = {"m", "male", "1"}

    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[s.isin({x.lower() for x in female_set})] = 1.0
    out[s.isin({x.lower() for x in male_set})] = 0.0
    return out


def mean_sd(series: pd.Series) -> Tuple[int, float, float]:
    s = clean_numeric(series).dropna().astype(float)
    n = int(len(s))
    if n == 0:
        return 0, np.nan, np.nan
    m = float(s.mean())
    sd = float(s.std(ddof=1)) if n > 1 else 0.0
    return n, m, sd


def fmt_mean_sd(m: float, sd: float, digits: int = 2) -> str:
    if pd.isna(m) or pd.isna(sd):
        return ""
    return f"{m:.{digits}f}±{sd:.{digits}f}"


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return ""
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def count_prop(binary01: pd.Series) -> Tuple[int, int, float]:
    b = pd.Series(binary01).dropna()
    if b.empty:
        return 0, 0, np.nan
    b = b.astype(int)
    n = int(len(b))
    x = int(b.sum())
    prop = float(b.mean())
    return n, x, prop


def fmt_count_pct(x: int, prop: float, digits: int = 2) -> str:
    if pd.isna(prop):
        return ""
    return f"{x:,} ({prop * 100:.{digits}f}%)"


def welch_ttest_pvalue(x: pd.Series, y: pd.Series) -> float:
    x = clean_numeric(pd.Series(x)).dropna().astype(float).values
    y = clean_numeric(pd.Series(y)).dropna().astype(float).values
    if len(x) < 2 or len(y) < 2:
        return np.nan
    try:
        from scipy import stats

        _, p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
        return float(p)
    except Exception:
        return np.nan


def chi2_pvalue_2x2(a: int, b: int, c: int, d: int) -> float:
    try:
        from scipy import stats

        table = np.array([[a, b], [c, d]], dtype=float)
        _, p, _, _ = stats.chi2_contingency(table, correction=False)
        return float(p)
    except Exception:
        return np.nan


def fisher_exact_pvalue_2x2(a: int, b: int, c: int, d: int) -> float:
    try:
        from scipy import stats

        table = np.array([[a, b], [c, d]], dtype=int)
        _, p = stats.fisher_exact(table, alternative="two-sided")
        return float(p)
    except Exception:
        return np.nan


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_table_weight_loss(df_h: pd.DataFrame, df_e: pd.DataFrame) -> pd.DataFrame:
    col_age = pick_first_existing(df_h, ["age"]) or "age"
    col_sex = pick_first_existing(df_h, ["sex"]) or "sex"
    col_bmi = pick_case_insensitive(df_h, ["bmi", "BMI"]) or "bmi"

    baseline_weight_candidates = ["baseline_weight_kg", "baseline_weight", "weight_baseline"]
    col_w0_h = pick_first_existing(df_h, baseline_weight_candidates)
    col_w0_e = pick_first_existing(df_e, baseline_weight_candidates)
    if col_w0_h is None or col_w0_e is None:
        raise ValueError(
            "Baseline weight column not found in both arms. "
            f"Candidates={baseline_weight_candidates}. "
            f"Human_found={col_w0_h}, EPS_found={col_w0_e}"
        )

    df_h = df_h.copy()
    df_e = df_e.copy()
    df_h["Female01"] = clean_sex_binary_keepna(df_h[col_sex])
    df_e["Female01"] = clean_sex_binary_keepna(df_e[col_sex])
    df_all = pd.concat([df_h.assign(__group__="Human"), df_e.assign(__group__="EPS-human")], ignore_index=True)

    rows = []

    def add_row_cont(label: str, s_all: pd.Series, s_h: pd.Series, s_e: pd.Series) -> None:
        _, m_all, sd_all = mean_sd(s_all)
        _, m_h, sd_h = mean_sd(s_h)
        _, m_e, sd_e = mean_sd(s_e)
        p = welch_ttest_pvalue(s_e, s_h)
        rows.append(
            {
                "Characteristic": label,
                "Total": fmt_mean_sd(m_all, sd_all),
                "Human": fmt_mean_sd(m_h, sd_h),
                "EPS–human": fmt_mean_sd(m_e, sd_e),
                "P value": fmt_p(p),
            }
        )

    def add_row_binary(label: str, b_all: pd.Series, b_h: pd.Series, b_e: pd.Series) -> None:
        n_all, x_all, p_all = count_prop(b_all)
        n_h, x_h, p_h = count_prop(b_h)
        n_e, x_e, p_e = count_prop(b_e)

        # Pearson chi-square (2x2) as specified for Table 1
        a = x_h
        b = n_h - x_h
        c = x_e
        d = n_e - x_e
        pval = chi2_pvalue_2x2(a, b, c, d)

        rows.append(
            {
                "Characteristic": label,
                "Total": fmt_count_pct(x_all, p_all),
                "Human": fmt_count_pct(x_h, p_h),
                "EPS–human": fmt_count_pct(x_e, p_e),
                "P value": fmt_p(pval),
            }
        )

    rows.append(
        {
            "Characteristic": "No. of participants",
            "Total": f"n={len(df_all):,}",
            "Human": f"n={len(df_h):,}",
            "EPS–human": f"n={len(df_e):,}",
            "P value": "",
        }
    )

    add_row_cont("Age, years", df_all[col_age], df_h[col_age], df_e[col_age])
    add_row_binary("Female sex, n (%)", df_all["Female01"], df_h["Female01"], df_e["Female01"])
    add_row_cont("BMI", df_all[col_bmi], df_h[col_bmi], df_e[col_bmi])
    add_row_cont("Baseline weight (kg)", df_all[col_w0_h], df_h[col_w0_h], df_e[col_w0_e])

    return pd.DataFrame(rows, columns=["Characteristic", "Total", "Human", "EPS–human", "P value"])


def build_table_glycemic(df_h: pd.DataFrame, df_e: pd.DataFrame) -> pd.DataFrame:
    col_age = pick_first_existing(df_h, ["age"]) or "age"
    col_sex = pick_first_existing(df_h, ["sex"]) or "sex"
    col_bmi = pick_case_insensitive(df_h, ["bmi", "BMI"]) or "bmi"

    baseline_fpg_candidates = ["baseline_fpg_mmol", "baseline_fpg", "FPG0", "Baseline fasting glucose", "Fasting glucose"]
    col_fpg0_h = pick_first_existing(df_h, baseline_fpg_candidates)
    col_fpg0_e = pick_first_existing(df_e, baseline_fpg_candidates)
    if col_fpg0_h is None or col_fpg0_e is None:
        raise ValueError(
            "Baseline fasting glucose column not found in both arms. "
            f"Candidates={baseline_fpg_candidates}. "
            f"Human_found={col_fpg0_h}, EPS_found={col_fpg0_e}"
        )

    df_h = df_h.copy()
    df_e = df_e.copy()
    df_h["Female01"] = clean_sex_binary_keepna(df_h[col_sex])
    df_e["Female01"] = clean_sex_binary_keepna(df_e[col_sex])
    df_all = pd.concat([df_h.assign(__group__="Human"), df_e.assign(__group__="EPS-human")], ignore_index=True)

    rows = []

    def add_row_cont(label: str, s_all: pd.Series, s_h: pd.Series, s_e: pd.Series) -> None:
        _, m_all, sd_all = mean_sd(s_all)
        _, m_h, sd_h = mean_sd(s_h)
        _, m_e, sd_e = mean_sd(s_e)
        p = welch_ttest_pvalue(s_e, s_h)
        rows.append(
            {
                "Characteristic": label,
                "Total": fmt_mean_sd(m_all, sd_all),
                "Human": fmt_mean_sd(m_h, sd_h),
                "EPS–human": fmt_mean_sd(m_e, sd_e),
                "P value": fmt_p(p),
            }
        )

    def add_row_binary_fisher(label: str, b_all: pd.Series, b_h: pd.Series, b_e: pd.Series) -> None:
        n_all, x_all, p_all = count_prop(b_all)
        n_h, x_h, p_h = count_prop(b_h)
        n_e, x_e, p_e = count_prop(b_e)

        a = x_h
        b = n_h - x_h
        c = x_e
        d = n_e - x_e
        pval = fisher_exact_pvalue_2x2(a, b, c, d)

        rows.append(
            {
                "Characteristic": label,
                "Total": fmt_count_pct(x_all, p_all),
                "Human": fmt_count_pct(x_h, p_h),
                "EPS–human": fmt_count_pct(x_e, p_e),
                "P value": fmt_p(pval),
            }
        )

    rows.append(
        {
            "Characteristic": "No. of participants",
            "Total": f"n={len(df_all):,}",
            "Human": f"n={len(df_h):,}",
            "EPS–human": f"n={len(df_e):,}",
            "P value": "",
        }
    )

    add_row_cont("Age, years", df_all[col_age], df_h[col_age], df_e[col_age])
    add_row_binary_fisher("Female sex, n (%)", df_all["Female01"], df_h["Female01"], df_e["Female01"])
    add_row_cont("BMI", df_all[col_bmi], df_h[col_bmi], df_e[col_bmi])
    add_row_cont("Fasting glucose", df_all[col_fpg0_h], df_h[col_fpg0_h], df_e[col_fpg0_e])

    return pd.DataFrame(rows, columns=["Characteristic", "Total", "Human", "EPS–human", "P value"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline characteristics (Table 1).")
    p.add_argument("--weight_human", type=str, default=None, help="Path to Human weight-loss Excel.")
    p.add_argument("--weight_eps", type=str, default=None, help="Path to EPS-human weight-loss Excel.")
    p.add_argument("--gly_human", type=str, default=None, help="Path to Human glycemic-control Excel.")
    p.add_argument("--gly_eps", type=str, default=None, help="Path to EPS-human glycemic-control Excel.")
    p.add_argument("--out_dir", type=str, default="outputs/clinical_trial", help="Output directory.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    if args.weight_human is None or args.weight_eps is None or args.gly_human is None or args.gly_eps is None:
        raise SystemExit(
            "Missing input paths. Provide --weight_human --weight_eps --gly_human --gly_eps."
        )

    w_h = Path(args.weight_human)
    w_e = Path(args.weight_eps)
    g_h = Path(args.gly_human)
    g_e = Path(args.gly_eps)

    df_w_h = read_excel(w_h)
    df_w_e = read_excel(w_e)
    df_g_h = read_excel(g_h)
    df_g_e = read_excel(g_e)

    weight_table = build_table_weight_loss(df_w_h, df_w_e)
    glycemic_table = build_table_glycemic(df_g_h, df_g_e)

    out_xlsx = out_dir / "baseline_characteristics_tables.xlsx"
    ensure_parent_dir(out_xlsx)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        weight_table.to_excel(writer, index=False, sheet_name="Table1_weight_loss")
        glycemic_table.to_excel(writer, index=False, sheet_name="Table1_glycemic")

    weight_table.to_csv(out_dir / "Table1_weight_loss.csv", index=False)
    glycemic_table.to_csv(out_dir / "Table1_glycemic.csv", index=False)


if __name__ == "__main__":
    main()
