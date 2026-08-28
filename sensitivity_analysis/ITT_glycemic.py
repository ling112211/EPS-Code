"""
ITT Sensitivity Analysis — Glycemic Control Cohort (Exploratory)
=================================================================
Positioning:  EXPLORATORY supplementary sensitivity analysis reported in
              Supplementary Table 2.
              Glycemic cohort is small (24/24 randomized, 21/19 completers).
              Results should be presented in supplementary material only.

Population:   24 randomized per arm. Completers: Human=21, EPS=19.
              Missing: Human=3, EPS=5 — REAL baseline data used
              (from "glycemic control missing data" folder).

Workflow:
  1. Load completer data (Human/EPS glycemic-control.xlsx)
  2. Load REAL missing-participant baselines from the missing-data files
  3. Build ITT population (24+24=48) — completers + real missing baselines
  4. MICE imputation of endline FPG ONLY → derive change scores
  5. ANCOVA on each imputed dataset → pool via Rubin's rules
  6. Available-case ANCOVA (for comparison)
  7. BOCF (missing endline FPG = baseline FPG, i.e. 0 change)
  8. MI diagnostics

Note: MNAR delta-adjustment and tipping-point analyses are in tipping_point_analysis.py.

Requirements:
  pip install pandas numpy scipy openpyxl statsmodels scikit-learn
"""

import os
import pandas as pd
import numpy as np
import math
import warnings
# Suppress only convergence/optimization warnings from statsmodels; keep others visible
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", message=".*convergence.*", category=RuntimeWarning)

# ============================================================
# 0) Configuration — paths anchored to this script's directory
# ============================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
FILE_HUMAN      = os.path.join(_HERE, "..", "glycemic", "Human glycemic-control.xlsx")
FILE_EPS        = os.path.join(_HERE, "..", "glycemic", "EPS-Human glycemic-control.xlsx")
FILE_HUMAN_MISS = os.path.join(_HERE, "glycemic control missing data", "glycemic Human missing data.xlsx")
FILE_EPS_MISS   = os.path.join(_HERE, "glycemic control missing data", "glycemic EPS-human missing data.xlsx")
OUT_XLSX        = os.path.join(_HERE, "ITT_glycemic_results.xlsx")

N_RANDOMIZED_HUMAN = 24
N_RANDOMIZED_EPS   = 24
M_IMPUTATIONS = 20
SEED = 42

# ============================================================
# 1) Helper functions
# ============================================================
def clean_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def clean_sex(series):
    raw = series.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=raw.index, dtype=float)
    out[raw.isin({"女","female","f","0","2"})] = 1.0
    out[raw.isin({"男","male","m","1"})]       = 0.0
    return out

bw_candidates = ["初始体重（档案）"]

def pick_col(df, cands):
    for c in cands:
        if c in df.columns: return c
    return None

# ============================================================
# 2) Load COMPLETER data
# ============================================================
df_h = pd.read_excel(FILE_HUMAN)
df_e = pd.read_excel(FILE_EPS)
for df in [df_h, df_e]:
    df.columns = df.columns.astype(str).str.strip()

col_bw_h = pick_col(df_h, bw_candidates)
col_bw_e = pick_col(df_e, bw_candidates)

def prepare_glycemic(df, col_bw, label):
    d = pd.DataFrame({
        "age":    clean_numeric(df["年龄"]).values,
        "sex":    clean_sex(df["性别"]).values,
        "bmi":    clean_numeric(df["bmi"]).values,
        "bw":     clean_numeric(df[col_bw]).values if col_bw else np.nan,
        "fpg0":   clean_numeric(df["入营空腹"]).values,
        "fpg1":   clean_numeric(df["结营空腹"]).values,
    })
    d["group"] = label
    d["completer"] = 1
    d["fpg_change"]     = d["fpg0"] - d["fpg1"]
    d["fpg_change_pct"] = np.where(d["fpg0"] > 0,
                                    d["fpg_change"] / d["fpg0"] * 100, np.nan)
    return d

comp_h = prepare_glycemic(df_h, col_bw_h, "Human")
comp_e = prepare_glycemic(df_e, col_bw_e, "EPS")
print(f"Completers: Human={len(comp_h)}, EPS={len(comp_e)}")

# ============================================================
# 3) Load REAL missing-participant baselines
# ============================================================

# --- Human missing (3 rows, columns correctly aligned) ---
df_h_miss = pd.read_excel(FILE_HUMAN_MISS)
df_h_miss.columns = df_h_miss.columns.astype(str).str.strip()

def prepare_glycemic_missing_human(df):
    d = pd.DataFrame({
        "age":    clean_numeric(df["年龄"]).values,
        "sex":    clean_sex(df["性别"]).values,
        "bmi":    clean_numeric(df["bmi"]).values,
        "bw":     clean_numeric(df["初始体重（档案）"]).values,
        "fpg0":   clean_numeric(df["入营空腹"]).values,
        "fpg1":   np.nan,       # outcome unknown → to be imputed
        "fpg_change":     np.nan,
        "fpg_change_pct": np.nan,
    })
    d["group"]     = "Human"
    d["completer"] = 0
    return d

# --- EPS missing (5 rows, columns correctly aligned) ---
def prepare_glycemic_missing_eps(df):
    d = pd.DataFrame({
        "age":    clean_numeric(df["年龄"]).values,
        "sex":    clean_sex(df["性别"]).values,
        "bmi":    clean_numeric(df["bmi"]).values,
        "bw":     clean_numeric(df["初始体重（档案）"]).values,
        "fpg0":   clean_numeric(df["入营空腹"]).values,
        "fpg1":   np.nan,       # outcome unknown → to be imputed
        "fpg_change":     np.nan,
        "fpg_change_pct": np.nan,
    })
    d["group"]     = "EPS"
    d["completer"] = 0
    return d

df_e_miss = pd.read_excel(FILE_EPS_MISS)
df_e_miss.columns = df_e_miss.columns.astype(str).str.strip()

miss_h = prepare_glycemic_missing_human(df_h_miss)
miss_e = prepare_glycemic_missing_eps(df_e_miss)
print(f"Missing (real baselines): Human={len(miss_h)}, EPS={len(miss_e)}")

# Print loaded values for verification
print("  EPS missing baselines (age | sex | bw | bmi | fpg0):")
for _, r in miss_e.iterrows():
    print(f"    age={r.age:.0f} sex={r.sex} bw={r.bw:.1f} "
          f"bmi={r.bmi:.2f} fpg0={r.fpg0:.1f}")
print("  Human missing baselines:")
for _, r in miss_h.iterrows():
    print(f"    age={r.age:.0f} sex={r.sex} bw={r.bw:.1f} "
          f"bmi={r.bmi:.2f} fpg0={r.fpg0:.1f}")

# Verify counts
for arm, n_comp, n_miss, n_rand in [
    ("Human", len(comp_h), len(miss_h), N_RANDOMIZED_HUMAN),
    ("EPS",   len(comp_e), len(miss_e), N_RANDOMIZED_EPS),
]:
    total = n_comp + n_miss
    mark = "✓" if total == n_rand else "⚠"
    print(f"  {mark} {arm}: {n_comp}+{n_miss}={total} (N_RANDOMIZED={n_rand})")

# ============================================================
# 4) Build ITT population
# ============================================================
df_itt = pd.concat([comp_h, comp_e, miss_h, miss_e], ignore_index=True)
df_itt["group_num"] = (df_itt["group"] == "EPS").astype(float)

print(f"\nITT: N={len(df_itt)} (Human={sum(df_itt.group=='Human')}, EPS={sum(df_itt.group=='EPS')})")
print(f"Missing fpg1: {df_itt['fpg1'].isna().sum()}")

# ============================================================
# 5) MICE — impute endline FPG ONLY → derive change scores
# ============================================================
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.linear_model import BayesianRidge
from sklearn.impute import IterativeImputer
import statsmodels.api as sm
from scipy.stats import t as tdist

MI_COVARIATES = ["group_num", "age", "sex", "bmi", "fpg0"]
IMPUTE_COLS   = MI_COVARIATES + ["fpg1"]

def run_mice(df_full, m=M_IMPUTATIONS, seed=SEED):
    base = df_full[IMPUTE_COLS].copy()
    datasets = []
    for i in range(m):
        imp = IterativeImputer(
            estimator=BayesianRidge(), max_iter=50,
            random_state=seed + i, sample_posterior=True, skip_complete=True,
        )
        filled = pd.DataFrame(imp.fit_transform(base.values), columns=IMPUTE_COLS)
        dc = df_full.copy()
        dc["fpg1"] = filled["fpg1"]
        dc["fpg_change"]     = dc["fpg0"] - dc["fpg1"]
        dc["fpg_change_pct"] = np.where(dc["fpg0"] > 0,
                                         dc["fpg_change"] / dc["fpg0"] * 100, np.nan)
        datasets.append(dc)
    return datasets

print("\nRunning MICE (MAR)...")
imputed_mar = run_mice(df_itt)
print(f"  {len(imputed_mar)} imputed datasets.")

# ============================================================
# 6) ANCOVA + Rubin's rules
# ============================================================
def rubins_rules(estimates, variances):
    est = np.array(estimates, dtype=float)
    var = np.array(variances, dtype=float)
    valid = ~(np.isnan(est) | np.isnan(var))
    est, var = est[valid], var[valid]
    m = len(est)
    if m < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    Q = np.mean(est); U = np.mean(var); B = np.var(est, ddof=1)
    T = U + (1 + 1/m) * B
    se  = math.sqrt(T) if T > 0 else np.nan
    fmi = ((1 + 1/m) * B / T) if T > 0 else np.nan
    if B > 0 and U > 0:
        r = (1 + 1/m) * B / U; df = (m - 1) * (1 + 1/r)**2
    else:
        df = 1e6
    tcrit = float(tdist.ppf(0.975, df))
    ci_lo = Q - tcrit * se if not np.isnan(se) else np.nan
    ci_hi = Q + tcrit * se if not np.isnan(se) else np.nan
    p = 2 * float(tdist.sf(abs(Q / se), df)) if se and se > 0 else np.nan
    return Q, se, ci_lo, ci_hi, p, df, fmi

def ancova_effect(dc, outcome, covars=["fpg0", "age", "sex", "bmi"]):
    cols = ["group_num"] + covars + [outcome]
    d = dc[cols].dropna()
    if len(d) < len(cols) + 3:
        return np.nan, np.nan, np.nan, np.nan
    y = d[outcome].astype(float)
    X = sm.add_constant(d[["group_num"] + covars].astype(float))
    try:
        model = sm.OLS(y, X).fit()
        c = float(model.params["group_num"]); se = float(model.bse["group_num"])
        p = float(model.pvalues["group_num"])
        return c, se, se**2, p
    except Exception as e:
        warnings.warn(f"ANCOVA model failed: {e}")
        return np.nan, np.nan, np.nan, np.nan

def pool_ancova(datasets, outcome, covars=["fpg0","age","sex","bmi"]):
    coefs, variances = [], []
    eps_m, eps_v, hum_m, hum_v = [], [], [], []
    for dc in datasets:
        c, se, v, p = ancova_effect(dc, outcome, covars)
        coefs.append(c); variances.append(v)
        e = dc.loc[dc.group=="EPS",   outcome].dropna()
        h = dc.loc[dc.group=="Human", outcome].dropna()
        eps_m.append(e.mean()); eps_v.append((e.std(ddof=1)/math.sqrt(len(e)))**2 if len(e)>1 else np.nan)
        hum_m.append(h.mean()); hum_v.append((h.std(ddof=1)/math.sqrt(len(h)))**2 if len(h)>1 else np.nan)
    Q, se, lo, hi, p, df, fmi = rubins_rules(coefs, variances)
    eQ,_,eLo,eHi,_,_,_ = rubins_rules(eps_m, eps_v)
    hQ,_,hLo,hHi,_,_,_ = rubins_rules(hum_m, hum_v)
    return {
        "diff":     {"est": Q, "se": se, "lo": lo, "hi": hi, "p": p, "fmi": fmi},
        "eps_mean": {"est": eQ, "lo": eLo, "hi": eHi},
        "hum_mean": {"est": hQ, "lo": hLo, "hi": hHi},
    }

print("\n=== ITT — ANCOVA (MAR MI) ===")
res_fpg     = pool_ancova(imputed_mar, "fpg_change")
res_fpg_pct = pool_ancova(imputed_mar, "fpg_change_pct")

def fmt(v, d=3):
    return "" if pd.isna(v) else f"{v:.{d}f}"
def fmt_ci(e, lo, hi, d=3):
    return "" if any(pd.isna(x) for x in [e,lo,hi]) else f"{e:.{d}f} ({lo:.{d}f}, {hi:.{d}f})"
def fmt_p(p):
    return "" if pd.isna(p) else ("<0.0001" if p < 1e-4 else f"{p:.4f}")

for label, r in [("FPG change (mmol/L)", res_fpg), ("FPG change (%)", res_fpg_pct)]:
    d = r["diff"]
    print(f"  {label}: diff={fmt_ci(d['est'],d['lo'],d['hi'])}, p={fmt_p(d['p'])}, FMI={fmt(d['fmi'],2)}")

# ============================================================
# 7) Available-case ANCOVA
# ============================================================
print("\n=== Available-case — ANCOVA ===")
df_ac = pd.concat([comp_h, comp_e], ignore_index=True)
df_ac["group_num"] = (df_ac["group"] == "EPS").astype(float)

ac_results = {}
for outcome, label in [("fpg_change","FPG change"), ("fpg_change_pct","FPG change %")]:
    c, se, v, p = ancova_effect(df_ac, outcome)
    tcrit = float(tdist.ppf(0.975, max(len(df_ac)-6, 1)))
    ac_results[outcome] = {
        "diff": c, "lo": c-tcrit*se if not np.isnan(se) else np.nan,
        "hi":   c+tcrit*se if not np.isnan(se) else np.nan, "p": p,
        "n_eps": len(comp_e), "n_hum": len(comp_h),
        "eps_mean": comp_e[outcome].mean(), "hum_mean": comp_h[outcome].mean(),
    }
    print(f"  {label}: diff={fmt_ci(c, ac_results[outcome]['lo'], ac_results[outcome]['hi'])}, p={fmt_p(p)}")

# ============================================================
# 8) BOCF: missing endline FPG = baseline FPG (0 change)
# ============================================================
print("\n=== BOCF (missing = 0 FPG change) ===")
df_bocf = df_itt.copy()
miss_mask = df_bocf["completer"] == 0
df_bocf.loc[miss_mask, "fpg1"] = df_bocf.loc[miss_mask, "fpg0"]
df_bocf["fpg_change"]     = df_bocf["fpg0"] - df_bocf["fpg1"]
df_bocf["fpg_change_pct"] = np.where(df_bocf["fpg0"]>0,
                                      df_bocf["fpg_change"]/df_bocf["fpg0"]*100, np.nan)

bocf_results = {}
for outcome in ["fpg_change", "fpg_change_pct"]:
    c, se, v, p = ancova_effect(df_bocf, outcome)
    tcrit = float(tdist.ppf(0.975, max(len(df_bocf)-6, 1)))
    bocf_results[outcome] = {
        "diff": c, "lo": c-tcrit*se if not np.isnan(se) else np.nan,
        "hi":   c+tcrit*se if not np.isnan(se) else np.nan, "p": p,
    }
    print(f"  {outcome}: diff={fmt_ci(c, bocf_results[outcome]['lo'], bocf_results[outcome]['hi'])}, p={fmt_p(p)}")

# ============================================================
# 9) MI diagnostics
# ============================================================
obs_fpg1 = df_itt.loc[df_itt.completer==1, "fpg1"].dropna()
imp_fpg1s = []
for dc in imputed_mar:
    imp_fpg1s.extend(dc.loc[df_itt.completer==0, "fpg1"].dropna().values)
imp_fpg1 = np.array(imp_fpg1s)

diag_rows = [
    {"Metric": "Observed endline FPG — N",   "Value": len(obs_fpg1)},
    {"Metric": "Observed endline FPG — Mean", "Value": f"{obs_fpg1.mean():.3f}"},
    {"Metric": "Observed endline FPG — SD",   "Value": f"{obs_fpg1.std():.3f}"},
    {"Metric": "Imputed endline FPG — N",     "Value": f"{len(imp_fpg1)} (across {M_IMPUTATIONS} imps)"},
    {"Metric": "Imputed endline FPG — Mean",  "Value": f"{imp_fpg1.mean():.3f}" if len(imp_fpg1)>0 else "N/A"},
    {"Metric": "Imputed endline FPG — SD",    "Value": f"{imp_fpg1.std():.3f}"  if len(imp_fpg1)>0 else "N/A"},
    {"Metric": "FMI (fpg_change ANCOVA)",     "Value": fmt(res_fpg["diff"]["fmi"], 2)},
    {"Metric": "FMI (fpg_change_pct ANCOVA)", "Value": fmt(res_fpg_pct["diff"]["fmi"], 2)},
]

# ============================================================
# 10) Save
# ============================================================
rows_main = []
for outcome, label, r_itt, ac, bocf in [
    ("fpg_change",     "FPG reduction (mmol/L)", res_fpg,     ac_results["fpg_change"],     bocf_results["fpg_change"]),
    ("fpg_change_pct", "FPG reduction (%)",       res_fpg_pct, ac_results["fpg_change_pct"], bocf_results["fpg_change_pct"]),
]:
    d = r_itt["diff"]
    rows_main.append({
        "Outcome": label, "Analysis": "ITT (MI + ANCOVA, MAR)",
        "N_EPS": N_RANDOMIZED_EPS, "N_Human": N_RANDOMIZED_HUMAN,
        "EPS Mean": fmt(r_itt["eps_mean"]["est"]),
        "Human Mean": fmt(r_itt["hum_mean"]["est"]),
        "ANCOVA Adj Diff (95% CI)": fmt_ci(d["est"], d["lo"], d["hi"]),
        "P value": fmt_p(d["p"]), "FMI": fmt(d["fmi"], 2),
    })
    rows_main.append({
        "Outcome": label, "Analysis": "Available-case (ANCOVA)",
        "N_EPS": ac["n_eps"], "N_Human": ac["n_hum"],
        "EPS Mean": fmt(ac["eps_mean"]), "Human Mean": fmt(ac["hum_mean"]),
        "ANCOVA Adj Diff (95% CI)": fmt_ci(ac["diff"], ac["lo"], ac["hi"]),
        "P value": fmt_p(ac["p"]), "FMI": "",
    })
    rows_main.append({
        "Outcome": label, "Analysis": "BOCF (missing = baseline FPG)",
        "N_EPS": N_RANDOMIZED_EPS, "N_Human": N_RANDOMIZED_HUMAN,
        "EPS Mean": "", "Human Mean": "",
        "ANCOVA Adj Diff (95% CI)": fmt_ci(bocf["diff"], bocf["lo"], bocf["hi"]),
        "P value": fmt_p(bocf["p"]), "FMI": "",
    })

df_main = pd.DataFrame(rows_main)
df_diag = pd.DataFrame(diag_rows)

notes = pd.DataFrame([
    {"Item": "Positioning",        "Details": "EXPLORATORY supplementary sensitivity analysis"},
    {"Item": "Randomized",         "Details": f"{N_RANDOMIZED_HUMAN}+{N_RANDOMIZED_EPS}=48"},
    {"Item": "Completers",         "Details": f"Human={len(comp_h)}, EPS={len(comp_e)}"},
    {"Item": "Missing (real BL)",  "Details": f"Human={len(miss_h)}, EPS={len(miss_e)} — actual baseline data used"},
    {"Item": "Imputation target",  "Details": "Endline FPG (结营空腹) ONLY; change scores derived deterministically"},
    {"Item": "MI covariates",      "Details": ", ".join(MI_COVARIATES)},
    {"Item": "MI method",          "Details": "MICE + BayesianRidge, sample_posterior=True"},
    {"Item": "Analysis model",     "Details": "ANCOVA: fpg_change ~ group + fpg0 + age + sex + bmi"},
    {"Item": "BOCF",               "Details": "Missing endline FPG set to baseline FPG (0 change)"},
    {"Item": "MNAR / tipping point","Details": "See tipping_point_analysis.py"},
    {"Item": "Seed",               "Details": str(SEED)},
    {"Item": "Caution",            "Details": "Small sample (n=48); results should not be over-interpreted"},
])

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    df_main.to_excel(writer,  sheet_name="Main Results",   index=False)
    df_diag.to_excel(writer,  sheet_name="MI Diagnostics", index=False)
    notes.to_excel(writer,    sheet_name="Notes",          index=False)

print(f"\n✅ Saved to: {OUT_XLSX}")
