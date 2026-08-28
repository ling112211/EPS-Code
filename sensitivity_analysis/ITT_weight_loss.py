"""
ITT Sensitivity Analysis — Weight Loss Cohort
=================================================================
Positioning:  Primary ITT sensitivity analysis for the weight-loss cohort
              reported in Supplementary Table 1.

Population:   790 randomized per arm.
              Completers: Human=702, EPS=742.
              Non-completers: Human=88, EPS=48 — REAL baseline data now used
              (from "weight loss missing data" folder) instead of bootstrap resampling.

Workflow:
  1. Load completer data (Human weight-loss.xlsx, EPS-Human weight-loss.xlsx)
  2. Load REAL missing-participant baselines from the missing-data files
  3. Build ITT population (790+790=1580) — completers + real missing baselines
  4. MICE imputation of wl_kg for missing participants → derive wl_pct
  5. ANCOVA on each imputed dataset → pool via Rubin's rules
  6. Available-case ANCOVA (for comparison)
  7. BOCF (missing = 0 kg / 0 % weight loss)
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
FILE_HUMAN_COMP = os.path.join(_HERE, "..", "weight-loss", "Human weight-loss.xlsx")
FILE_EPS_COMP   = os.path.join(_HERE, "..", "weight-loss", "EPS-Human weight-loss.xlsx")
FILE_HUMAN_MISS = os.path.join(_HERE, "weight loss missing data", "weight loss human missing data.xlsx")
FILE_EPS_MISS   = os.path.join(_HERE, "weight loss missing data", "weight loss EPS-human missing data.xlsx")
OUT_XLSX        = os.path.join(_HERE, "ITT_weight_loss_results.xlsx")

N_RANDOMIZED_HUMAN = 790
N_RANDOMIZED_EPS   = 790
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
df_h_comp = pd.read_excel(FILE_HUMAN_COMP)
df_e_comp  = pd.read_excel(FILE_EPS_COMP)
for df in [df_h_comp, df_e_comp]:
    df.columns = df.columns.astype(str).str.strip()

def prepare_wl_completer(df, label):
    col_bw = pick_col(df, bw_candidates)
    d = pd.DataFrame({
        "age":         clean_numeric(df["年龄"]).values,
        "sex":         clean_sex(df["性别"]).values,
        "bmi":         clean_numeric(df["bmi"]).values,
        "baseline_wt": clean_numeric(df[col_bw]).values if col_bw else np.nan,
        "wl_kg":       clean_numeric(df["减重数"]).values,
    })
    d["group"] = label
    d["completer"] = 1
    d["wl_pct"] = np.where(d["baseline_wt"] > 0,
                            d["wl_kg"] / d["baseline_wt"] * 100, np.nan)
    return d

comp_h = prepare_wl_completer(df_h_comp, "Human")
comp_e = prepare_wl_completer(df_e_comp, "EPS")
print(f"Completers: Human={len(comp_h)}, EPS={len(comp_e)}")

# ============================================================
# 3) Load REAL missing-participant baselines
# ============================================================
df_h_miss = pd.read_excel(FILE_HUMAN_MISS)
df_e_miss  = pd.read_excel(FILE_EPS_MISS)
for df in [df_h_miss, df_e_miss]:
    df.columns = df.columns.astype(str).str.strip()

def prepare_wl_missing(df, label):
    d = pd.DataFrame({
        "age":         clean_numeric(df["年龄"]).values,
        "sex":         clean_sex(df["性别"]).values,
        "bmi":         clean_numeric(df["bmi"]).values,
        "baseline_wt": clean_numeric(df["初始体重（档案）"]).values,
        "wl_kg":       np.nan,   # outcome unknown → to be imputed
        "wl_pct":      np.nan,
    })
    d["group"] = label
    d["completer"] = 0
    return d

miss_h = prepare_wl_missing(df_h_miss, "Human")
miss_e = prepare_wl_missing(df_e_miss, "EPS")
print(f"Missing (real baselines): Human={len(miss_h)}, EPS={len(miss_e)}")

# Verify counts
n_total_h = len(comp_h) + len(miss_h)
n_total_e = len(comp_e) + len(miss_e)
if n_total_h != N_RANDOMIZED_HUMAN:
    print(f"  ⚠ Human: {len(comp_h)} completers + {len(miss_h)} missing = {n_total_h} ≠ {N_RANDOMIZED_HUMAN}")
else:
    print(f"  ✓ Human: {len(comp_h)} + {len(miss_h)} = {n_total_h} = N_RANDOMIZED")
if n_total_e != N_RANDOMIZED_EPS:
    print(f"  ⚠ EPS:   {len(comp_e)} completers + {len(miss_e)} missing = {n_total_e} ≠ {N_RANDOMIZED_EPS}")
else:
    print(f"  ✓ EPS:   {len(comp_e)} + {len(miss_e)} = {n_total_e} = N_RANDOMIZED")

# ============================================================
# 4) Build ITT population
# ============================================================
df_itt = pd.concat([comp_h, comp_e, miss_h, miss_e], ignore_index=True)
df_itt["group_num"] = (df_itt["group"] == "EPS").astype(float)

print(f"\nITT: N={len(df_itt)} (Human={sum(df_itt.group=='Human')}, EPS={sum(df_itt.group=='EPS')})")
print(f"Missing wl_kg: {df_itt['wl_kg'].isna().sum()}")
print(f"Completer rate: Human={sum((df_itt.group=='Human') & (df_itt.completer==1))}/{sum(df_itt.group=='Human')}, "
      f"EPS={sum((df_itt.group=='EPS') & (df_itt.completer==1))}/{sum(df_itt.group=='EPS')}")

# ============================================================
# 5) MICE — impute wl_kg for missing participants → re-derive wl_pct
# ============================================================
from sklearn.experimental import enable_iterative_imputer   # noqa
from sklearn.linear_model import BayesianRidge
from sklearn.impute import IterativeImputer
import statsmodels.api as sm
from scipy.stats import t as tdist

IMPUTE_COLS = ["group_num", "age", "sex", "bmi", "baseline_wt", "wl_kg"]

def run_mice(df_full, m=M_IMPUTATIONS, seed=SEED):
    """MICE: impute wl_kg → re-derive wl_pct."""
    base = df_full[IMPUTE_COLS].copy()
    datasets = []
    for i in range(m):
        imp = IterativeImputer(
            estimator=BayesianRidge(), max_iter=50,
            random_state=seed + i, sample_posterior=True, skip_complete=True,
        )
        filled = pd.DataFrame(imp.fit_transform(base.values), columns=IMPUTE_COLS)
        dc = df_full.copy()
        dc["wl_kg"] = filled["wl_kg"]
        bw = filled["baseline_wt"]
        dc["wl_pct"] = np.where(bw > 0, dc["wl_kg"] / bw * 100, np.nan)
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

    Q = np.mean(est)
    U = np.mean(var)
    B = np.var(est, ddof=1)
    T = U + (1 + 1/m) * B
    se = math.sqrt(T) if T > 0 else np.nan
    fmi = ((1 + 1/m) * B / T) if T > 0 else np.nan

    if B > 0 and U > 0:
        r = (1 + 1/m) * B / U
        df = (m - 1) * (1 + 1/r)**2
    else:
        df = 1e6

    tcrit = float(tdist.ppf(0.975, df))
    ci_lo = Q - tcrit * se if not np.isnan(se) else np.nan
    ci_hi = Q + tcrit * se if not np.isnan(se) else np.nan
    p = 2 * float(tdist.sf(abs(Q / se), df)) if se and se > 0 else np.nan
    return Q, se, ci_lo, ci_hi, p, df, fmi

def ancova_effect(dc, outcome, covars=["baseline_wt", "age", "sex", "bmi"]):
    cols = ["group_num"] + covars + [outcome]
    d = dc[cols].dropna()
    if len(d) < len(cols) + 3:
        return np.nan, np.nan, np.nan, np.nan
    y = d[outcome].astype(float)
    X = sm.add_constant(d[["group_num"] + covars].astype(float))
    try:
        model = sm.OLS(y, X).fit()
        c = float(model.params["group_num"])
        se = float(model.bse["group_num"])
        p = float(model.pvalues["group_num"])
        return c, se, se**2, p
    except Exception as e:
        warnings.warn(f"ANCOVA model failed: {e}")
        return np.nan, np.nan, np.nan, np.nan

def pool_ancova(datasets, outcome, covars=["baseline_wt", "age", "sex", "bmi"]):
    coefs, variances = [], []
    eps_m, eps_v, hum_m, hum_v = [], [], [], []
    for dc in datasets:
        c, se, v, p = ancova_effect(dc, outcome, covars)
        coefs.append(c); variances.append(v)
        e = dc.loc[dc.group == "EPS", outcome].dropna()
        h = dc.loc[dc.group == "Human", outcome].dropna()
        eps_m.append(e.mean())
        eps_v.append((e.std(ddof=1)/math.sqrt(len(e)))**2 if len(e) > 1 else np.nan)
        hum_m.append(h.mean())
        hum_v.append((h.std(ddof=1)/math.sqrt(len(h)))**2 if len(h) > 1 else np.nan)

    Q, se, lo, hi, p, df, fmi = rubins_rules(coefs, variances)
    eQ,_,eLo,eHi,_,_,_ = rubins_rules(eps_m, eps_v)
    hQ,_,hLo,hHi,_,_,_ = rubins_rules(hum_m, hum_v)
    return {
        "diff":     {"est": Q, "se": se, "lo": lo, "hi": hi, "p": p, "fmi": fmi},
        "eps_mean": {"est": eQ, "lo": eLo, "hi": eHi},
        "hum_mean": {"est": hQ, "lo": hLo, "hi": hHi},
    }

# Run ITT ANCOVA
print("\n=== ITT — ANCOVA (MAR MI) ===")
res_kg  = pool_ancova(imputed_mar, "wl_kg")
res_pct = pool_ancova(imputed_mar, "wl_pct")

def fmt(v, d=3):
    return "" if pd.isna(v) else f"{v:.{d}f}"
def fmt_ci(e, lo, hi, d=3):
    return "" if any(pd.isna(x) for x in [e, lo, hi]) else f"{e:.{d}f} ({lo:.{d}f}, {hi:.{d}f})"
def fmt_p(p):
    return "" if pd.isna(p) else ("<0.0001" if p < 1e-4 else f"{p:.4f}")

for label, r in [("WL (kg)", res_kg), ("WL (%)", res_pct)]:
    d = r["diff"]
    print(f"  {label}: diff={fmt_ci(d['est'],d['lo'],d['hi'])}, p={fmt_p(d['p'])}, FMI={fmt(d['fmi'],2)}")

# ============================================================
# 7) Available-case ANCOVA
# ============================================================
print("\n=== Available-case — ANCOVA ===")
df_ac = pd.concat([comp_h, comp_e], ignore_index=True)
df_ac["group_num"] = (df_ac["group"] == "EPS").astype(float)

ac_results = {}
for outcome, label in [("wl_kg","WL (kg)"), ("wl_pct","WL (%)")]:
    c, se, v, p = ancova_effect(df_ac, outcome)
    tcrit = float(tdist.ppf(0.975, max(len(df_ac)-6, 1)))
    ac_results[outcome] = {
        "diff": c,
        "lo":   c - tcrit * se if not np.isnan(se) else np.nan,
        "hi":   c + tcrit * se if not np.isnan(se) else np.nan,
        "p":    p,
        "n_eps": len(comp_e), "n_hum": len(comp_h),
        "eps_mean": comp_e[outcome].mean(), "hum_mean": comp_h[outcome].mean(),
    }
    print(f"  {label}: diff={fmt_ci(c, ac_results[outcome]['lo'], ac_results[outcome]['hi'])}, p={fmt_p(p)}")

# ============================================================
# 8) BOCF: missing = 0 kg / 0 % weight loss
# ============================================================
print("\n=== BOCF (missing = 0 kg weight loss) ===")
df_bocf = df_itt.copy()
miss_mask = df_bocf["completer"] == 0
df_bocf.loc[miss_mask, "wl_kg"]  = 0.0
df_bocf.loc[miss_mask, "wl_pct"] = 0.0

bocf_results = {}
for outcome in ["wl_kg", "wl_pct"]:
    c, se, v, p = ancova_effect(df_bocf, outcome)
    tcrit = float(tdist.ppf(0.975, max(len(df_bocf)-6, 1)))
    bocf_results[outcome] = {
        "diff": c,
        "lo":   c - tcrit * se if not np.isnan(se) else np.nan,
        "hi":   c + tcrit * se if not np.isnan(se) else np.nan,
        "p":    p,
    }
    print(f"  {outcome}: diff={fmt_ci(c, bocf_results[outcome]['lo'], bocf_results[outcome]['hi'])}, p={fmt_p(p)}")

# ============================================================
# 9) Modified Poisson — binary responder endpoints (≥2% and ≥5% WL)
# ============================================================
print("\n=== Modified Poisson — Binary Responders ===")

def add_responder_cols(datasets):
    """Create binary ≥2% and ≥5% WL columns in each imputed dataset."""
    out = []
    for dc in datasets:
        dc2 = dc.copy()
        dc2["resp_2pct"] = np.where(dc2["wl_pct"].isna(), np.nan,
                                     (dc2["wl_pct"] >= 2).astype(float))
        dc2["resp_5pct"] = np.where(dc2["wl_pct"].isna(), np.nan,
                                     (dc2["wl_pct"] >= 5).astype(float))
        out.append(dc2)
    return out

def modified_poisson_effect(dc, outcome, covars=["baseline_wt", "age", "sex", "bmi"]):
    """Modified Poisson with robust (HC1) SE → log(RR) and its variance."""
    cols = ["group_num"] + covars + [outcome]
    d = dc[cols].dropna()
    if len(d) < len(cols) + 3 or d[outcome].sum() < 3:
        return np.nan, np.nan
    y = d[outcome].astype(float)
    X = sm.add_constant(d[["group_num"] + covars].astype(float))
    try:
        model = sm.GLM(y, X, family=sm.families.Poisson()).fit(cov_type="HC1")
        log_rr = float(model.params["group_num"])
        var_log = float(model.cov_params().loc["group_num", "group_num"])
        return log_rr, var_log
    except Exception as e:
        warnings.warn(f"Modified Poisson model failed: {e}")
        return np.nan, np.nan

def pool_poisson(datasets, outcome, covars=["baseline_wt", "age", "sex", "bmi"]):
    """Pool Modified Poisson via Rubin's rules on log-RR scale."""
    log_rrs, variances = [], []
    eps_props, hum_props = [], []
    for dc in datasets:
        log_rr, var = modified_poisson_effect(dc, outcome, covars)
        log_rrs.append(log_rr)
        variances.append(var)
        eps_props.append(dc.loc[dc["group"] == "EPS",    outcome].mean())
        hum_props.append(dc.loc[dc["group"] == "Human",  outcome].mean())

    Q, se, lo, hi, p, df, fmi = rubins_rules(log_rrs, variances)
    rr    = np.exp(Q)  if not np.isnan(Q)  else np.nan
    rr_lo = np.exp(lo) if not np.isnan(lo) else np.nan
    rr_hi = np.exp(hi) if not np.isnan(hi) else np.nan
    return {
        "rr": rr, "rr_lo": rr_lo, "rr_hi": rr_hi, "p": p, "fmi": fmi,
        "eps_prop": float(np.nanmean(eps_props)),
        "hum_prop": float(np.nanmean(hum_props)),
    }

imputed_resp = add_responder_cols(imputed_mar)
res_2pct = pool_poisson(imputed_resp, "resp_2pct")
res_5pct = pool_poisson(imputed_resp, "resp_5pct")

for label, r in [("≥2% WL responders", res_2pct), ("≥5% WL responders", res_5pct)]:
    print(f"  {label}: RR={r['rr']:.3f} ({r['rr_lo']:.3f}, {r['rr_hi']:.3f}), "
          f"p={fmt_p(r['p'])}, FMI={fmt(r['fmi'],2)}")
    print(f"    EPS prop={r['eps_prop']:.3f}, Human prop={r['hum_prop']:.3f}")

# ============================================================
# 11) MI diagnostics
# ============================================================
obs_wl = df_itt.loc[df_itt.completer == 1, "wl_kg"].dropna()
imp_wls = []
for dc in imputed_mar:
    imp_wls.extend(dc.loc[df_itt.completer == 0, "wl_kg"].dropna().values)
imp_wl = np.array(imp_wls)

diag_rows = [
    {"Metric": "Observed wl_kg — N",    "Value": len(obs_wl)},
    {"Metric": "Observed wl_kg — Mean", "Value": f"{obs_wl.mean():.3f}"},
    {"Metric": "Observed wl_kg — SD",   "Value": f"{obs_wl.std():.3f}"},
    {"Metric": "Imputed wl_kg — N",     "Value": f"{len(imp_wl)} (across {M_IMPUTATIONS} imps)"},
    {"Metric": "Imputed wl_kg — Mean",  "Value": f"{imp_wl.mean():.3f}" if len(imp_wl) > 0 else "N/A"},
    {"Metric": "Imputed wl_kg — SD",    "Value": f"{imp_wl.std():.3f}"  if len(imp_wl) > 0 else "N/A"},
    {"Metric": "FMI (wl_kg ANCOVA)",    "Value": fmt(res_kg["diff"]["fmi"], 2)},
    {"Metric": "FMI (wl_pct ANCOVA)",   "Value": fmt(res_pct["diff"]["fmi"], 2)},
]

# ============================================================
# 12) Save
# ============================================================
rows_main = []
for outcome, label, r_itt, ac, bocf in [
    ("wl_kg",  "Weight loss (kg)", res_kg,  ac_results["wl_kg"],  bocf_results["wl_kg"]),
    ("wl_pct", "Weight loss (%)",  res_pct, ac_results["wl_pct"], bocf_results["wl_pct"]),
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
        "EPS Mean": fmt(ac["eps_mean"]),
        "Human Mean": fmt(ac["hum_mean"]),
        "ANCOVA Adj Diff (95% CI)": fmt_ci(ac["diff"], ac["lo"], ac["hi"]),
        "P value": fmt_p(ac["p"]), "FMI": "",
    })
    rows_main.append({
        "Outcome": label, "Analysis": "BOCF (missing = 0 kg loss)",
        "N_EPS": N_RANDOMIZED_EPS, "N_Human": N_RANDOMIZED_HUMAN,
        "EPS Mean": "", "Human Mean": "",
        "ANCOVA Adj Diff (95% CI)": fmt_ci(bocf["diff"], bocf["lo"], bocf["hi"]),
        "P value": fmt_p(bocf["p"]), "FMI": "",
    })

# Modified Poisson rows
rows_poisson = []
for label, r in [("≥2% weight loss", res_2pct), ("≥5% weight loss", res_5pct)]:
    rows_poisson.append({
        "Outcome":    label,
        "Analysis":   "ITT (Modified Poisson, MAR MI)",
        "N_EPS":      N_RANDOMIZED_EPS,
        "N_Human":    N_RANDOMIZED_HUMAN,
        "EPS Prop":   fmt(r["eps_prop"], 3),
        "Human Prop": fmt(r["hum_prop"], 3),
        "RR (95% CI)": (f"{r['rr']:.3f} ({r['rr_lo']:.3f}, {r['rr_hi']:.3f})"
                        if not any(np.isnan(x) for x in [r["rr"], r["rr_lo"], r["rr_hi"]]) else ""),
        "P value":    fmt_p(r["p"]),
        "FMI":        fmt(r["fmi"], 2),
    })

df_main    = pd.DataFrame(rows_main)
df_poisson = pd.DataFrame(rows_poisson)
df_diag    = pd.DataFrame(diag_rows)

notes = pd.DataFrame([
    {"Item": "Positioning",       "Details": "Primary ITT analysis — weight-loss cohort"},
    {"Item": "Randomized",        "Details": f"{N_RANDOMIZED_HUMAN}+{N_RANDOMIZED_EPS}={N_RANDOMIZED_HUMAN+N_RANDOMIZED_EPS}"},
    {"Item": "Completers",        "Details": f"Human={len(comp_h)}, EPS={len(comp_e)}"},
    {"Item": "Missing (real BL)", "Details": f"Human={len(miss_h)}, EPS={len(miss_e)} — actual baseline CRF data used"},
    {"Item": "Missing data source","Details": "weight loss missing data/weight loss human/EPS-human missing data.xlsx"},
    {"Item": "Imputation target", "Details": "wl_kg; wl_pct re-derived from wl_kg/baseline_wt"},
    {"Item": "MI covariates",     "Details": ", ".join(IMPUTE_COLS)},
    {"Item": "MI method",         "Details": "MICE + BayesianRidge, sample_posterior=True"},
    {"Item": "Analysis model",    "Details": "ANCOVA: outcome ~ group + baseline_wt + age + sex + bmi"},
    {"Item": "BOCF",              "Details": "Missing participants assigned 0 kg / 0% weight loss"},
    {"Item": "MNAR/tipping point","Details": "See tipping_point_analysis.py"},
    {"Item": "Seed",              "Details": str(SEED)},
])

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    df_main.to_excel(writer,    sheet_name="Main Results",      index=False)
    df_poisson.to_excel(writer, sheet_name="Responder Analysis", index=False)
    df_diag.to_excel(writer,    sheet_name="MI Diagnostics",    index=False)
    notes.to_excel(writer,      sheet_name="Notes",             index=False)

print(f"\n✅ Saved to: {OUT_XLSX}")
