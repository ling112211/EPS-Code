"""
Tipping-Point Sensitivity Analysis
====================================
Core question: "How much worse would missing outcomes in the EPS arm
need to be (relative to MAR imputation) before the EPS vs Human
treatment effect loses statistical significance?"

Primary scenario: EPS-arm-specific delta (most conservative for EPS benefit).
Secondary: differential delta (EPS vs Human arms).

Reports both: p > 0.05 crossing AND CI crossing zero. These tipping-point
rows are incorporated into Supplementary Tables 1 and 2.

★ This script is the SOLE source of MNAR delta-adjustment and tipping-point
  results. The ITT scripts (ITT_weight_loss.py, ITT_glycemic.py) handle
  MAR MI, available-case, BOCF, and diagnostics only.

★ Implementation details (column names, covariates, MI parameters, ANCOVA
  specification) are aligned with the ITT scripts to ensure consistency.

Requirements:
  pip install pandas numpy scipy openpyxl statsmodels scikit-learn
"""

import os
import pandas as pd
import numpy as np
import math
import re
import warnings
# Suppress only convergence/optimization warnings from statsmodels; keep others visible
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", message=".*convergence.*", category=RuntimeWarning)

# ============================================================
# 0) Configuration — paths anchored to this script's directory
# ============================================================
_HERE = os.path.dirname(os.path.abspath(__file__))

# Weight loss — completer files
FILE_EPS_WL   = os.path.join(_HERE, "..", "weight-loss", "EPS-Human weight-loss.xlsx")
FILE_HUMAN_WL = os.path.join(_HERE, "..", "weight-loss", "Human weight-loss.xlsx")
# Weight loss — REAL missing baseline files (replaces bootstrap)
FILE_HUMAN_WL_MISS = os.path.join(_HERE, "weight loss missing data", "weight loss human missing data.xlsx")
FILE_EPS_WL_MISS   = os.path.join(_HERE, "weight loss missing data", "weight loss EPS-human missing data.xlsx")
N_RAND_HUMAN_WL = 790
N_RAND_EPS_WL   = 790

# Glycemic — completer files
FILE_HUMAN_GL = os.path.join(_HERE, "..", "glycemic", "Human glycemic-control.xlsx")
FILE_EPS_GL   = os.path.join(_HERE, "..", "glycemic", "EPS-Human glycemic-control.xlsx")
# Glycemic — REAL missing baseline files (replaces bootstrap)
FILE_HUMAN_GL_MISS = os.path.join(_HERE, "glycemic control missing data", "glycemic Human missing data.xlsx")
FILE_EPS_GL_MISS   = os.path.join(_HERE, "glycemic control missing data", "glycemic EPS-human missing data.xlsx")
N_RAND_HUMAN_GL = 24
N_RAND_EPS_GL   = 24

OUT_XLSX = os.path.join(_HERE, "tipping_point_results.xlsx")
M_IMPUTATIONS = 20
SEED = 42

# ============================================================
# 1) Shared helpers
# ============================================================
def clean_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def clean_sex(series):
    raw = series.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=raw.index, dtype=float)
    out[raw.isin({"女","female","f","0","2"})] = 1.0
    out[raw.isin({"男","male","m","1"})]       = 0.0
    return out

def clean_ratio_percent(series):
    s = series.astype(str).str.strip()
    pat = re.compile(r"^[\+\-]?\s*\d*\.?\d+\s*%?\s*$")
    valid = s.apply(lambda x: bool(pat.match(x)))
    out = pd.Series(np.nan, index=series.index, dtype=float)
    if valid.any():
        tmp = s[valid].str.replace("%","",regex=False).str.replace(" ","",regex=False).astype(float)
        if len(tmp) > 0 and tmp.abs().max() < 1:
            tmp *= 100
        out.loc[tmp.index] = tmp
    return out

def pick_col(df, cands):
    for c in cands:
        if c in df.columns: return c
    return None

import statsmodels.api as sm
from scipy.stats import t as tdist

def ancova_effect(dc, outcome, covars):
    """ANCOVA: outcome ~ group + covariates.
    Returns (coef, se, p) for the group effect."""
    cols = ["group_num"] + covars + [outcome]
    d = dc[cols].dropna()
    if len(d) < len(cols) + 3:
        return np.nan, np.nan, np.nan
    y = d[outcome].astype(float)
    X = sm.add_constant(d[["group_num"]+covars].astype(float))
    try:
        m = sm.OLS(y, X).fit()
        return float(m.params["group_num"]), float(m.bse["group_num"]), float(m.pvalues["group_num"])
    except Exception as e:
        warnings.warn(f"ANCOVA model failed: {e}")
        return np.nan, np.nan, np.nan

def rubins_rules(estimates, variances):
    est = np.array(estimates, dtype=float)
    var = np.array(variances, dtype=float)
    ok = ~(np.isnan(est)|np.isnan(var))
    est, var = est[ok], var[ok]
    m = len(est)
    if m < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    Q = np.mean(est); U = np.mean(var); B = np.var(est, ddof=1)
    T = U + (1+1/m)*B
    se = math.sqrt(T) if T > 0 else np.nan
    if B > 0 and U > 0:
        r = (1+1/m)*B/U; df = (m-1)*(1+1/r)**2
    else:
        df = 1e6
    tcrit = float(tdist.ppf(0.975, df))
    lo = Q - tcrit*se if not np.isnan(se) else np.nan
    hi = Q + tcrit*se if not np.isnan(se) else np.nan
    p = 2*float(tdist.sf(abs(Q/se), df)) if se and se > 0 else np.nan
    return Q, lo, hi, p, se

from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import BayesianRidge
from sklearn.impute import IterativeImputer

# ============================================================
# 2) Weight Loss: build ITT population
#    ★ Aligned with ITT_weight_loss.py: column name "baseline_wt",
#      covariates ["baseline_wt","age","sex","bmi"],
#      MICE includes ["group_num","age","sex","bmi","baseline_wt","wl_kg"]
# ============================================================
print("=" * 60)
print("TIPPING-POINT SENSITIVITY ANALYSIS")
print("=" * 60)

df_e_wl = pd.read_excel(FILE_EPS_WL)
df_h_wl = pd.read_excel(FILE_HUMAN_WL)
for df in [df_e_wl, df_h_wl]:
    df.columns = df.columns.astype(str).str.strip()

bw_cands = ["初始体重（档案）"]
col_bw_e = pick_col(df_e_wl, bw_cands)
col_bw_h = pick_col(df_h_wl, bw_cands)

def prep_wl(df, col_bw, label):
    d = pd.DataFrame({
        "age": clean_numeric(df["年龄"]).values,
        "sex": clean_sex(df["性别"]).values,
        "bmi": clean_numeric(df["bmi"]).values,
        "baseline_wt": clean_numeric(df[col_bw]).values if col_bw else np.nan,
        "wl_kg": clean_numeric(df["减重数"]).values,
    })
    d["group"] = label; d["completer"] = 1
    d["wl_pct"] = np.where(d["baseline_wt"]>0, d["wl_kg"]/d["baseline_wt"]*100, np.nan)
    return d

comp_e_wl = prep_wl(df_e_wl, col_bw_e, "EPS")
comp_h_wl = prep_wl(df_h_wl, col_bw_h, "Human")

# ★ Load REAL missing-participant baselines (replaces bootstrap resampling)
df_h_wl_miss = pd.read_excel(FILE_HUMAN_WL_MISS)
df_e_wl_miss = pd.read_excel(FILE_EPS_WL_MISS)
for df in [df_h_wl_miss, df_e_wl_miss]:
    df.columns = df.columns.astype(str).str.strip()

def prep_wl_missing(df, label):
    d = pd.DataFrame({
        "age":         clean_numeric(df["年龄"]).values,
        "sex":         clean_sex(df["性别"]).values,
        "bmi":         clean_numeric(df["bmi"]).values,
        "baseline_wt": clean_numeric(df["初始体重（档案）"]).values,
        "wl_kg":       np.nan,
        "wl_pct":      np.nan,
    })
    d["group"] = label; d["completer"] = 0
    return d

miss_h_wl = prep_wl_missing(df_h_wl_miss, "Human")
miss_e_wl = prep_wl_missing(df_e_wl_miss, "EPS")

n_miss_h_wl = N_RAND_HUMAN_WL - len(comp_h_wl)
n_miss_e_wl = N_RAND_EPS_WL   - len(comp_e_wl)

df_itt_wl = pd.concat([comp_h_wl, comp_e_wl, miss_h_wl, miss_e_wl], ignore_index=True)
df_itt_wl["group_num"] = (df_itt_wl["group"]=="EPS").astype(float)

print(f"WL ITT: N={len(df_itt_wl)} (Human={sum(df_itt_wl.group=='Human')}, EPS={sum(df_itt_wl.group=='EPS')})")

# ============================================================
# 3) Glycemic: build ITT population
#    ★ Aligned with ITT_glycemic.py: covariates ["fpg0","age","sex","bmi"]
# ============================================================
df_h_gl = pd.read_excel(FILE_HUMAN_GL)
df_e_gl = pd.read_excel(FILE_EPS_GL)
for df in [df_h_gl, df_e_gl]:
    df.columns = df.columns.astype(str).str.strip()

def prep_gl(df, label):
    col_bw = pick_col(df, ["初始体重（档案）"])
    d = pd.DataFrame({
        "age":  clean_numeric(df["年龄"]).values,
        "sex":  clean_sex(df["性别"]).values,
        "bmi":  clean_numeric(df["bmi"]).values,
        "bw":   clean_numeric(df[col_bw]).values if col_bw else np.nan,
        "fpg0": clean_numeric(df["入营空腹"]).values,
        "fpg1": clean_numeric(df["结营空腹"]).values,
    })
    d["group"] = label; d["completer"] = 1
    d["fpg_change"] = d["fpg0"] - d["fpg1"]
    return d

comp_h_gl = prep_gl(df_h_gl, "Human")
comp_e_gl = prep_gl(df_e_gl, "EPS")

# ★ Load REAL missing-participant baselines for glycemic (replaces bootstrap)
df_h_gl_miss = pd.read_excel(FILE_HUMAN_GL_MISS)
df_h_gl_miss.columns = df_h_gl_miss.columns.astype(str).str.strip()

def prep_gl_missing_human(df):
    d = pd.DataFrame({
        "age":  clean_numeric(df["年龄"]).values,
        "sex":  clean_sex(df["性别"]).values,
        "bmi":  clean_numeric(df["bmi"]).values,
        "bw":   clean_numeric(df["初始体重（档案）"]).values,
        "fpg0": clean_numeric(df["入营空腹"]).values,
        "fpg1": np.nan,
        "fpg_change": np.nan,
    })
    d["group"] = "Human"; d["completer"] = 0
    return d

def prep_gl_missing_eps(df):
    """EPS glycemic missing — columns correctly aligned."""
    d = pd.DataFrame({
        "age":  clean_numeric(df["年龄"]).values,
        "sex":  clean_sex(df["性别"]).values,
        "bmi":  clean_numeric(df["bmi"]).values,
        "bw":   clean_numeric(df["初始体重（档案）"]).values,
        "fpg0": clean_numeric(df["入营空腹"]).values,
        "fpg1": np.nan,
        "fpg_change": np.nan,
    })
    d["group"] = "EPS"; d["completer"] = 0
    return d

df_e_gl_miss = pd.read_excel(FILE_EPS_GL_MISS)
df_e_gl_miss.columns = df_e_gl_miss.columns.astype(str).str.strip()

miss_h_gl = prep_gl_missing_human(df_h_gl_miss)
miss_e_gl = prep_gl_missing_eps(df_e_gl_miss)

n_miss_h_gl = N_RAND_HUMAN_GL - len(comp_h_gl)
n_miss_e_gl = N_RAND_EPS_GL   - len(comp_e_gl)

df_itt_gl = pd.concat([comp_h_gl, comp_e_gl, miss_h_gl, miss_e_gl], ignore_index=True)
df_itt_gl["group_num"] = (df_itt_gl["group"]=="EPS").astype(float)

print(f"GL ITT: N={len(df_itt_gl)} (Human={sum(df_itt_gl.group=='Human')}, EPS={sum(df_itt_gl.group=='EPS')})")

# ============================================================
# 4) MICE functions
#    ★ Aligned with respective ITT scripts
# ============================================================
def mice_wl(df_full, m=M_IMPUTATIONS, seed=SEED):
    """MICE for WL — aligned with ITT_weight_loss.py.
    Impute cols: group_num, age, sex, bmi, baseline_wt, wl_kg."""
    cols = ["group_num","age","sex","bmi","baseline_wt","wl_kg"]
    base = df_full[cols].copy()
    out = []
    for i in range(m):
        imp = IterativeImputer(estimator=BayesianRidge(), max_iter=50,
            random_state=seed+i, sample_posterior=True, skip_complete=True)
        filled = pd.DataFrame(imp.fit_transform(base.values), columns=cols)
        dc = df_full.copy()
        dc["wl_kg"] = filled["wl_kg"]
        bw = filled["baseline_wt"]
        dc["wl_pct"] = np.where(bw>0, dc["wl_kg"]/bw*100, np.nan)
        out.append(dc)
    return out

def mice_gl(df_full, m=M_IMPUTATIONS, seed=SEED):
    """MICE for GL — aligned with ITT_glycemic.py.
    Impute cols: group_num, age, sex, bmi, fpg0, fpg1."""
    cols = ["group_num","age","sex","bmi","fpg0","fpg1"]
    base = df_full[cols].copy()
    out = []
    for i in range(m):
        imp = IterativeImputer(estimator=BayesianRidge(), max_iter=50,
            random_state=seed+i, sample_posterior=True, skip_complete=True)
        filled = pd.DataFrame(imp.fit_transform(base.values), columns=cols)
        dc = df_full.copy()
        dc["fpg1"] = filled["fpg1"]
        dc["fpg_change"] = dc["fpg0"] - dc["fpg1"]
        out.append(dc)
    return out

# ============================================================
# 5) Tipping-point search — ANCOVA-based
# ============================================================
def tipping_search(df_itt, mice_func, outcome_col, delta_range, covars,
                    apply_to="eps_only", delta_target="wl_kg", m=M_IMPUTATIONS, seed=SEED):
    """
    Search delta values. For weight-loss, delta applied to wl_kg → re-derive wl_pct.
    For glycemic, delta applied to fpg1 → re-derive fpg_change.

    Returns list of dicts with (delta, estimate, ci_lo, ci_hi, p, significant, ci_excludes_zero).
    """
    base_datasets = mice_func(df_itt, m=m, seed=seed)
    results = []

    for delta in delta_range:
        coefs, vars_ = [], []
        for dc in base_datasets:
            ds = dc.copy()
            miss = df_itt["completer"] == 0
            if apply_to == "eps_only":
                mask = miss & (df_itt["group"] == "EPS")
            elif apply_to == "human_only":
                mask = miss & (df_itt["group"] == "Human")
            else:
                mask = miss

            # Apply delta to the fundamental variable, re-derive
            if delta_target == "wl_kg":
                ds.loc[mask, "wl_kg"] += delta
                bw = ds["baseline_wt"]
                ds["wl_pct"] = np.where(bw>0, ds["wl_kg"]/bw*100, np.nan)
            elif delta_target == "fpg1":
                ds.loc[mask, "fpg1"] += delta  # positive = worse endline
                ds["fpg_change"] = ds["fpg0"] - ds["fpg1"]

            c, se, p = ancova_effect(ds, outcome_col, covars)
            coefs.append(c)
            vars_.append(se**2 if not np.isnan(se) else np.nan)

        Q, lo, hi, p, se = rubins_rules(coefs, vars_)
        sig = p < 0.05 if not np.isnan(p) else None
        ci_excl = (lo > 0 or hi < 0) if not (np.isnan(lo) or np.isnan(hi)) else None

        results.append({
            "delta": delta, "estimate": Q, "ci_lo": lo, "ci_hi": hi,
            "p": p, "significant": sig, "ci_excludes_zero": ci_excl
        })

    return results

def find_tipping(results):
    """Find delta where significance first lost (p>0.05 OR CI includes 0).

    Returns (delta, p, lo, hi, already_nonsig):
      already_nonsig=True  → result was non-significant at delta=0 (MAR baseline);
                             no MNAR perturbation is needed to nullify it.
      already_nonsig=False → result was significant at delta=0 and tipped at delta.
      delta=None           → result remained significant across all tested deltas.
    """
    for i, r in enumerate(results):
        if r["significant"] is False or r["ci_excludes_zero"] is False:
            return r["delta"], r["p"], r["ci_lo"], r["ci_hi"], (i == 0)
    return None, None, None, None, False

# ============================================================
# 6) Run tipping-point analyses
#    ★ Covariates aligned with ITT scripts
# ============================================================
wl_covars = ["baseline_wt","age","sex","bmi"]  # ← aligned with ITT_weight_loss.py
gl_covars = ["fpg0","age","sex","bmi"]          # ← aligned with ITT_glycemic.py

# --- Weight loss ---
print("\n--- Weight Loss Tipping Points ---")
wl_delta_range = np.arange(0, -10.1, -0.5)

print("  A) EPS dropouts only → wl_kg outcome...")
tp_wl_eps_kg = tipping_search(df_itt_wl, mice_wl, "wl_kg", wl_delta_range, wl_covars,
                               "eps_only", "wl_kg")

print("  B) EPS dropouts only → wl_pct outcome...")
tp_wl_eps_pct = tipping_search(df_itt_wl, mice_wl, "wl_pct", wl_delta_range, wl_covars,
                                "eps_only", "wl_kg")

print("  C) Differential: EPS worse, Human better (±delta)...")
# Custom: EPS dropouts get -delta (worse), Human dropouts get +delta (better)
tp_wl_diff = []
base_wl = mice_wl(df_itt_wl, m=M_IMPUTATIONS, seed=SEED)
for delta in wl_delta_range:
    coefs, vars_ = [], []
    for dc in base_wl:
        ds = dc.copy()
        miss = df_itt_wl["completer"] == 0
        eps_miss = miss & (df_itt_wl["group"] == "EPS")
        hum_miss = miss & (df_itt_wl["group"] == "Human")
        ds.loc[eps_miss, "wl_kg"] += delta        # EPS worse
        ds.loc[hum_miss, "wl_kg"] -= delta        # Human better
        bw = ds["baseline_wt"]
        ds["wl_pct"] = np.where(bw>0, ds["wl_kg"]/bw*100, np.nan)
        c, se, p = ancova_effect(ds, "wl_kg", wl_covars)
        coefs.append(c); vars_.append(se**2 if not np.isnan(se) else np.nan)
    Q, lo, hi, p, se = rubins_rules(coefs, vars_)
    tp_wl_diff.append({
        "delta": delta, "estimate": Q, "ci_lo": lo, "ci_hi": hi, "p": p,
        "significant": p < 0.05 if not np.isnan(p) else None,
        "ci_excludes_zero": (lo>0 or hi<0) if not (np.isnan(lo) or np.isnan(hi)) else None
    })

# --- Glycemic ---
print("\n--- Glycemic Tipping Points ---")
gl_delta_range = np.arange(0, 5.1, 0.2)  # positive = higher endline FPG = worse

print("  D) EPS dropouts only → fpg_change outcome...")
tp_gl_eps = tipping_search(df_itt_gl, mice_gl, "fpg_change", gl_delta_range, gl_covars,
                            "eps_only", "fpg1")

# ============================================================
# 7) Summary
# ============================================================
print("\n" + "="*60)
print("TIPPING POINT SUMMARY")
print("="*60)

analyses = [
    ("WL (kg) — EPS dropouts worse",       tp_wl_eps_kg,  "kg"),
    ("WL (%) — EPS dropouts worse",        tp_wl_eps_pct, "kg (applied to wl_kg)"),
    ("WL (kg) — Differential (EPS worse, Human better)", tp_wl_diff, "kg each direction"),
    ("FPG change — EPS dropouts worse",     tp_gl_eps,     "mmol/L on endline FPG"),
]

summary_rows = []
for name, res, unit in analyses:
    d, p, lo, hi, already_nonsig = find_tipping(res)
    if already_nonsig:
        # MAR baseline result is already non-significant — no tipping point applies
        print(f"  {name}: ★ already non-significant at MAR baseline (delta=0), "
              f"p={p:.4f}, CI=({lo:.3f}, {hi:.3f})")
        summary_rows.append({
            "Analysis": name, "Tipping delta": "N/A (baseline non-sig)", "Unit": unit,
            "P at tipping": f"{p:.4f}", "CI at tipping": f"({lo:.3f}, {hi:.3f})",
            "Interpretation": (
                "MAR ITT result already non-significant (p>0.05); "
                "MNAR sensitivity not applicable — no EPS benefit to erode."
            )
        })
    elif d is not None:
        print(f"  {name}: tipping at delta={d:.1f} {unit}, p={p:.4f}, CI=({lo:.3f}, {hi:.3f})")
        summary_rows.append({
            "Analysis": name, "Tipping delta": f"{d:.1f}", "Unit": unit,
            "P at tipping": f"{p:.4f}", "CI at tipping": f"({lo:.3f}, {hi:.3f})",
            "Interpretation": f"Result non-significant when EPS dropouts are {abs(d):.1f} {unit} worse than MAR"
        })
    else:
        print(f"  {name}: robust across all deltas tested (up to {res[-1]['delta']:.1f})")
        summary_rows.append({
            "Analysis": name, "Tipping delta": f"Beyond {res[-1]['delta']:.1f}", "Unit": unit,
            "P at tipping": "N/A", "CI at tipping": "N/A",
            "Interpretation": "Robust: significant across all tested deltas"
        })

# ============================================================
# 8) Save
# ============================================================
df_summary = pd.DataFrame(summary_rows)

def fmt_p(p):
    return "" if pd.isna(p) else ("<0.0001" if p < 1e-4 else f"{p:.4f}")

# Full detail sheets
def detail_df(results, label):
    rows = []
    for r in results:
        rows.append({
            "Delta": r["delta"],
            "ANCOVA Estimate (EPS-Human)": f"{r['estimate']:.4f}" if not np.isnan(r['estimate']) else "",
            "CI low": f"{r['ci_lo']:.4f}" if not np.isnan(r['ci_lo']) else "",
            "CI high": f"{r['ci_hi']:.4f}" if not np.isnan(r['ci_hi']) else "",
            "P value": fmt_p(r["p"]),
            "Significant (p<0.05)": r["significant"],
            "CI excludes 0": r["ci_excludes_zero"],
        })
    return pd.DataFrame(rows)

df_wl_kg  = detail_df(tp_wl_eps_kg,  "WL kg — EPS dropouts")
df_wl_pct = detail_df(tp_wl_eps_pct, "WL % — EPS dropouts")
df_wl_dif = detail_df(tp_wl_diff,    "WL kg — differential")
df_gl     = detail_df(tp_gl_eps,     "FPG — EPS dropouts")

notes = pd.DataFrame([
    {"Item": "Purpose",      "Details": "How far from MAR must missing data be to nullify the EPS benefit?"},
    {"Item": "Primary scenario", "Details": "Delta applied to EPS-arm dropouts only (most conservative)"},
    {"Item": "Differential scenario", "Details": "EPS dropouts worse by delta, Human dropouts better by delta (extreme)"},
    {"Item": "WL delta unit", "Details": "kg, applied to wl_kg; wl_pct re-derived from wl_kg/baseline_wt"},
    {"Item": "Glycemic delta unit", "Details": "mmol/L, applied to endline FPG; fpg_change re-derived"},
    {"Item": "Tipping criterion", "Details": "p > 0.05 OR 95% CI includes zero"},
    {"Item": "WL analysis model", "Details": "ANCOVA: outcome ~ group + baseline_wt + age + sex + bmi (aligned with ITT_weight_loss.py)"},
    {"Item": "GL analysis model", "Details": "ANCOVA: outcome ~ group + fpg0 + age + sex + bmi (aligned with ITT_glycemic.py)"},
    {"Item": "Imputations per delta", "Details": str(M_IMPUTATIONS)},
    {"Item": "Seed", "Details": str(SEED)},
    {"Item": "Missing data source", "Details": "Real baseline CRF data used for non-completers (weight loss missing data / glycemic control missing data folders)"},
    {"Item": "Consistency note", "Details": "Column names, covariates, and MI parameters match the ITT scripts exactly"},
])

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    df_summary.to_excel(writer, sheet_name="Summary", index=False)
    df_wl_kg.to_excel(writer,  sheet_name="WL kg EPS-only", index=False)
    df_wl_pct.to_excel(writer, sheet_name="WL pct EPS-only", index=False)
    df_wl_dif.to_excel(writer, sheet_name="WL kg Differential", index=False)
    df_gl.to_excel(writer,     sheet_name="FPG EPS-only", index=False)
    notes.to_excel(writer,     sheet_name="Notes", index=False)

print(f"\n✅ Saved to: {OUT_XLSX}")
