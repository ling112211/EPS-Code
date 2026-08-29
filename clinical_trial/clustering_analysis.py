#!/usr/bin/env python3
"""ICC and clustered analyses for the two clinical trial cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import openpyxl
import pandas as pd
import scipy
import statsmodels
import statsmodels.formula.api as smf
from scipy import optimize, stats


@dataclass(frozen=True)
class CohortSpec:
    name: str
    outcome: str
    effect_label: str
    baseline_columns: tuple[str, ...]
    endline_columns: tuple[str, ...]
    direct_outcome_columns: tuple[str, ...] = ()
    require_shared_manager_set: bool = False
    profile_community_icc: bool = True


@dataclass
class MixedFit:
    result: Any
    method: str
    attempts: list[dict[str, Any]]


@dataclass
class ICCProfile:
    point: float
    lower: float
    upper: float
    method: str
    status: str


SPECS = {
    "weight_loss": CohortSpec(
        name="weight_loss",
        outcome="weight_loss_kg",
        effect_label="EPS-human minus Human weight loss (kg)",
        baseline_columns=("入营体重", "baseline_weight_kg", "entry_weight_kg"),
        endline_columns=("出营体重", "endline_weight_kg", "exit_weight_kg"),
        direct_outcome_columns=("减重数", "weight_loss_kg", "weight_loss"),
        require_shared_manager_set=True,
        profile_community_icc=False,
    ),
    "glycemic": CohortSpec(
        name="glycemic",
        outcome="fpg_reduction_mmol_l",
        effect_label="EPS-human minus Human fasting glucose reduction (mmol/L)",
        baseline_columns=("入营空腹", "baseline_fpg_mmol_l", "baseline_fpg"),
        endline_columns=("结营空腹", "endline_fpg_mmol_l", "endline_fpg"),
    ),
}

ID_COLUMNS = ("序号", "participant_id", "id")
COMMUNITY_COLUMNS = ("班级号", "community_id", "class_id")
MANAGER_COLUMNS = ("健管师 id", "健管师id", "manager_id", "health_manager_id")
OPTIMIZERS = ("lbfgs", "powell", "cg", "nm", "bfgs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate ICCs and clustered treatment effects for a trial cohort."
    )
    parser.add_argument("--cohort", required=True, choices=sorted(SPECS))
    parser.add_argument("--human-xlsx", required=True, type=Path)
    parser.add_argument("--eps-xlsx", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    return parser.parse_args()


def find_column(frame: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"Missing {label} column. Accepted names: {', '.join(candidates)}")


def normalize_manager_id(series: pd.Series, source: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        rows = np.flatnonzero(numeric.isna().to_numpy())[:10].tolist()
        raise ValueError(f"{source}: missing or nonnumeric manager IDs at rows {rows}")
    if not np.allclose(numeric.to_numpy(), np.round(numeric.to_numpy())):
        raise ValueError(f"{source}: manager IDs must be integer-valued")
    return numeric.round().astype(int).astype(str)


def read_arm(path: Path, arm: str, spec: CohortSpec) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    source = pd.read_excel(path, sheet_name=0)
    source.columns = source.columns.astype(str).str.strip()
    participant_column = find_column(source, ID_COLUMNS, "participant ID")
    community_column = find_column(source, COMMUNITY_COLUMNS, "community ID")
    manager_column = find_column(source, MANAGER_COLUMNS, "health manager ID")
    baseline_column = find_column(source, spec.baseline_columns, "baseline outcome")
    endline_column = find_column(source, spec.endline_columns, "endline outcome")

    data = pd.DataFrame(index=source.index)
    data["participant_id"] = source[participant_column].astype(str).str.strip()
    data["arm"] = arm
    data["arm_eps"] = int(arm == "EPS-human")
    data["manager_id"] = normalize_manager_id(source[manager_column], path.name)
    data["community_label"] = source[community_column].astype(str).str.strip()
    data["community_id"] = arm + "::" + data["community_label"]
    data["baseline_value"] = pd.to_numeric(source[baseline_column], errors="coerce")
    data["endline_value"] = pd.to_numeric(source[endline_column], errors="coerce")
    calculated = data["baseline_value"] - data["endline_value"]
    direct_column = next(
        (column for column in spec.direct_outcome_columns if column in source.columns),
        None,
    )
    data[spec.outcome] = (
        pd.to_numeric(source[direct_column], errors="coerce")
        if direct_column is not None
        else calculated
    )
    data["calculated_change"] = calculated
    data["source_file"] = path.name
    return data.reset_index(drop=True)


def check_row(
    check: str,
    observed: Any,
    expected: Any,
    passed: bool,
    critical: bool = False,
) -> dict[str, Any]:
    return {
        "check": check,
        "observed": observed,
        "expected": expected,
        "passed": bool(passed),
        "critical": bool(critical),
    }


def validate_data(data: pd.DataFrame, spec: CohortSpec) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    missing = int(data[spec.outcome].isna().sum())
    rows.append(check_row("Missing primary outcomes", missing, 0, missing == 0, True))
    missing_measurements = int(
        data[["baseline_value", "endline_value"]].isna().any(axis=1).sum()
    )
    rows.append(
        check_row(
            "Rows missing baseline or endline values",
            missing_measurements,
            0,
            missing_measurements == 0,
            True,
        )
    )
    unique_ids = int(data["participant_id"].nunique(dropna=False))
    rows.append(
        check_row(
            "Unique participant IDs across both files",
            unique_ids,
            len(data),
            unique_ids == len(data),
            True,
        )
    )
    identity_error = float(
        np.nanmax(np.abs(data[spec.outcome] - data["calculated_change"]))
    )
    rows.append(
        check_row(
            "Outcome equals baseline minus endline",
            f"maximum absolute difference {identity_error:.10g}",
            "<= 1e-8",
            identity_error <= 1e-8,
            True,
        )
    )

    manager_sets = {
        arm: set(group["manager_id"])
        for arm, group in data.groupby("arm", observed=True)
    }
    eps_managers = manager_sets.get("EPS-human", set())
    human_managers = manager_sets.get("Human", set())
    manager_union = eps_managers | human_managers
    manager_intersection = eps_managers & human_managers
    rows.extend(
        [
            check_row(
                "Managers represented in pooled complete cases",
                len(manager_union),
                ">= 3",
                len(manager_union) >= 3,
                True,
            ),
            check_row(
                "Managers represented in both arms",
                len(manager_intersection),
                ">= 1",
                len(manager_intersection) >= 1,
                True,
            ),
        ]
    )
    if spec.require_shared_manager_set:
        rows.append(
            check_row(
                "Manager ID sets are identical across arms",
                eps_managers == human_managers,
                True,
                eps_managers == human_managers,
                True,
            )
        )

    communities_by_arm = (
        data.groupby("arm", observed=True)["community_id"].nunique().to_dict()
    )
    rows.append(
        check_row(
            "Communities represented by arm",
            json.dumps(communities_by_arm, sort_keys=True),
            ">= 2 per arm",
            len(communities_by_arm) == 2
            and all(value >= 2 for value in communities_by_arm.values()),
            True,
        )
    )
    manager_sizes = data.groupby("manager_id", observed=True).size()
    community_sizes = data.groupby("community_id", observed=True).size()
    managers_per_community = data.groupby("community_id", observed=True)[
        "manager_id"
    ].nunique()
    communities_per_manager_arm = data.groupby(
        ["manager_id", "arm"], observed=True
    )["community_id"].nunique()
    arms_per_manager = data.groupby("manager_id", observed=True)["arm"].nunique()
    rows.extend(
        [
            check_row(
                "Participant count per manager",
                f"{int(manager_sizes.min())}-{int(manager_sizes.max())}",
                "Descriptive",
                True,
            ),
            check_row(
                "Singleton manager clusters",
                int((manager_sizes == 1).sum()),
                "Descriptive",
                True,
            ),
            check_row(
                "Participant count per community",
                f"{int(community_sizes.min())}-{int(community_sizes.max())}",
                "Descriptive",
                True,
            ),
            check_row(
                "Managers represented per community",
                f"{int(managers_per_community.min())}-{int(managers_per_community.max())}",
                "Descriptive",
                True,
            ),
            check_row(
                "Communities represented per manager within an arm",
                f"{int(communities_per_manager_arm.min())}-{int(communities_per_manager_arm.max())}",
                "Descriptive",
                True,
            ),
            check_row(
                "Trial arms represented per manager",
                f"{int(arms_per_manager.min())}-{int(arms_per_manager.max())}",
                "Descriptive",
                True,
            ),
            check_row(
                "Observed outcome range",
                f"{data[spec.outcome].min():.3f} to {data[spec.outcome].max():.3f}",
                "Descriptive review",
                True,
            ),
        ]
    )
    checks = pd.DataFrame(rows)
    failures = checks.loc[checks["critical"] & ~checks["passed"], "check"].tolist()
    if failures:
        raise ValueError("Critical data checks failed: " + "; ".join(failures))
    return checks


def fit_mixed_model(
    model_factory: Callable[[], Any],
    methods: Iterable[str] = OPTIMIZERS,
    disagreement_threshold: float = 1e-3,
) -> MixedFit:
    attempts: list[dict[str, Any]] = []
    candidates: list[tuple[float, str, Any, int]] = []
    for method in methods:
        index = len(attempts)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = model_factory().fit(
                    reml=True, method=method, maxiter=3000, disp=False
                )
            llf = float(result.llf)
            finite = bool(
                np.isfinite(llf)
                and np.all(np.isfinite(np.asarray(result.params, dtype=float)))
                and np.isfinite(result.scale)
                and result.scale > 0
            )
            valid = bool(result.converged) and finite
            attempts.append(
                {
                    "method": method,
                    "converged": bool(result.converged),
                    "finite_solution": finite,
                    "valid_candidate": valid,
                    "llf": llf,
                    "arm_estimate": (
                        float(result.fe_params["arm_eps"])
                        if "arm_eps" in result.fe_params.index
                        else None
                    ),
                    "selected": False,
                    "warnings": " | ".join(str(item.message) for item in caught),
                }
            )
            if valid:
                candidates.append((llf, method, result, index))
        except Exception as exc:
            attempts.append(
                {
                    "method": method,
                    "converged": False,
                    "finite_solution": False,
                    "valid_candidate": False,
                    "llf": None,
                    "arm_estimate": None,
                    "selected": False,
                    "warnings": f"{type(exc).__name__}: {exc}",
                }
            )
    if not candidates:
        raise RuntimeError(f"No valid converged mixed-model solution: {attempts}")

    best_llf, best_method, best_result, best_index = max(
        candidates, key=lambda item: item[0]
    )
    valid_llfs = sorted((item[0] for item in candidates), reverse=True)
    spread = best_llf - min(valid_llfs)
    second_gap = best_llf - valid_llfs[1] if len(valid_llfs) > 1 else math.nan
    disagreement = len(valid_llfs) > 1 and spread > disagreement_threshold
    for attempt in attempts:
        llf = attempt["llf"]
        attempt["llf_gap_to_best"] = (
            float(best_llf - llf)
            if attempt["valid_candidate"] and llf is not None
            else None
        )
        attempt["best_vs_second_llf_gap"] = (
            float(second_gap) if np.isfinite(second_gap) else None
        )
        attempt["converged_llf_spread"] = float(spread)
        attempt["optimizer_disagreement_warning"] = disagreement
    attempts[best_index]["selected"] = True
    attempts[best_index]["selection_warning"] = (
        f"Converged solutions differed by {spread:.6g} log-likelihood units."
        if disagreement
        else ""
    )
    return MixedFit(best_result, best_method, attempts)


def random_intercept_variance(result: Any) -> float:
    return max(0.0, float(result.cov_re.iloc[0, 0]))


def variance_on_boundary(random_variance: float, residual_variance: float) -> bool:
    denominator = max(random_variance + residual_variance, np.finfo(float).tiny)
    return bool(random_variance / denominator < 1e-8)


def profile_icc_ci(
    data: pd.DataFrame, cluster_column: str, outcome_column: str, alpha: float = 0.05
) -> ICCProfile:
    y = data[outcome_column].to_numpy(dtype=float)
    clusters = data[cluster_column].astype(str).to_numpy()
    z = np.column_stack([(clusters == value) for value in pd.unique(clusters)]).astype(
        float
    )
    zz = z @ z.T
    x = np.ones((len(data), 1), dtype=float)
    n, p = x.shape

    def reml_loglike(log_ratio: float) -> float:
        covariance = np.eye(n) + float(np.exp(log_ratio)) * zz
        sign_v, logdet_v = np.linalg.slogdet(covariance)
        if sign_v <= 0:
            return -np.inf
        inverse = np.linalg.inv(covariance)
        information = x.T @ inverse @ x
        sign_i, logdet_i = np.linalg.slogdet(information)
        if sign_i <= 0:
            return -np.inf
        beta = np.linalg.solve(information, x.T @ inverse @ y)
        residual_sum = float((y - x @ beta).T @ inverse @ (y - x @ beta))
        scale = residual_sum / (n - p)
        if scale <= 0 or not np.isfinite(scale):
            return -np.inf
        return float(
            -0.5
            * ((n - p) * (np.log(2 * np.pi) + 1 + np.log(scale)) + logdet_v + logdet_i)
        )

    lower_bound = -30.0
    optimum = optimize.minimize_scalar(
        lambda value: -reml_loglike(value),
        bounds=(lower_bound, 20.0),
        method="bounded",
        options={"xatol": 1e-12},
    )
    if not optimum.success or not np.isfinite(optimum.fun):
        return ICCProfile(
            math.nan,
            math.nan,
            math.nan,
            "Direct REML profile over the variance ratio",
            "profile_optimization_failed",
        )
    boundary_llf = reml_loglike(lower_bound)
    optimum_log_ratio = float(optimum.x)
    optimum_llf = reml_loglike(optimum_log_ratio)
    if boundary_llf >= optimum_llf - 1e-9:
        optimum_log_ratio = lower_bound
        maximum_llf = boundary_llf
        point = 0.0
    else:
        maximum_llf = optimum_llf
        ratio = float(np.exp(optimum_log_ratio))
        point = ratio / (1 + ratio)
    cutoff = maximum_llf - float(stats.chi2.ppf(1 - alpha, 1) / 2)
    if boundary_llf >= cutoff:
        lower = 0.0
    else:
        root = optimize.brentq(
            lambda value: reml_loglike(value) - cutoff,
            lower_bound,
            optimum_log_ratio,
        )
        ratio = float(np.exp(root))
        lower = ratio / (1 + ratio)
    upper_bound = max(optimum_log_ratio + 1, -5.0)
    while upper_bound < 50 and reml_loglike(upper_bound) > cutoff:
        upper_bound += 1
    if upper_bound >= 50 and reml_loglike(upper_bound) > cutoff:
        return ICCProfile(
            point,
            lower,
            math.nan,
            "Direct REML profile over the variance ratio",
            "upper_limit_not_bracketed",
        )
    root = optimize.brentq(
        lambda value: reml_loglike(value) - cutoff,
        optimum_log_ratio,
        upper_bound,
    )
    ratio = float(np.exp(root))
    return ICCProfile(
        point,
        lower,
        ratio / (1 + ratio),
        "Direct REML profile likelihood over tau^2 / sigma^2",
        "ok",
    )


def unavailable_profile(point: float, status: str) -> ICCProfile:
    return ICCProfile(point, math.nan, math.nan, "Not calculated", status)


def base_result_row(spec: CohortSpec) -> dict[str, Any]:
    return {
        "cohort": spec.name,
        "effect": spec.effect_label,
        "random_variance_boundary": np.nan,
        "boundary_gls_substitution": False,
        "manager_random_effect_included": False,
        "community_random_effect_included": False,
        "manager_variance": np.nan,
        "community_variance": np.nan,
        "residual_variance": np.nan,
        "manager_icc_conditional": np.nan,
        "community_icc_conditional": np.nan,
        "fit_attempts": "",
    }


def extract_arm_effect(
    fit: MixedFit,
    spec: CohortSpec,
    model: str,
    role: str,
    clustering: str,
    manager_variance: Optional[float],
    community_variance: Optional[float],
) -> dict[str, Any]:
    result = fit.result
    estimate = float(result.fe_params["arm_eps"])
    residual_variance = float(result.scale)
    manager_included = manager_variance is not None
    community_included = community_variance is not None
    manager_value = max(0.0, float(manager_variance)) if manager_included else np.nan
    community_value = (
        max(0.0, float(community_variance)) if community_included else np.nan
    )
    random_variance = sum(
        value for value in (manager_value, community_value) if np.isfinite(value)
    )
    boundary = variance_on_boundary(random_variance, residual_variance)
    design = np.asarray(result.model.exog, dtype=float)
    arm_position = list(result.model.exog_names).index("arm_eps")
    if boundary:
        covariance = residual_variance * np.linalg.inv(design.T @ design)
        standard_error = float(np.sqrt(covariance[arm_position, arm_position]))
        se_method = "Plug-in GLS at zero variance-component boundary"
    else:
        standard_error = float(result.bse_fe["arm_eps"])
        se_method = "Mixed-model covariance from selected REML fit"
    statistic = estimate / standard_error
    critical = float(stats.norm.ppf(0.975))
    total_variance = residual_variance + random_variance
    row = base_result_row(spec)
    row.update(
        {
            "model": model,
            "reporting_role": role,
            "clustering": clustering,
            "estimate": estimate,
            "standard_error": standard_error,
            "ci_95_lower": estimate - critical * standard_error,
            "ci_95_upper": estimate + critical * standard_error,
            "test_statistic": statistic,
            "reference_distribution": "Standard normal (Wald)",
            "degrees_of_freedom": np.nan,
            "p_value": float(2 * stats.norm.sf(abs(statistic))),
            "fixed_effect_se_method": se_method,
            "random_variance_boundary": boundary,
            "boundary_gls_substitution": boundary,
            "manager_random_effect_included": manager_included,
            "community_random_effect_included": community_included,
            "manager_variance": manager_value,
            "community_variance": community_value,
            "residual_variance": residual_variance,
            "manager_icc_conditional": (
                manager_value / total_variance if manager_included else np.nan
            ),
            "community_icc_conditional": (
                community_value / total_variance if community_included else np.nan
            ),
            "converged": bool(result.converged),
            "optimizer": fit.method,
            "fit_attempts": json.dumps(fit.attempts, ensure_ascii=False),
        }
    )
    return row


def welch_row(data: pd.DataFrame, spec: CohortSpec) -> dict[str, Any]:
    eps = data.loc[data["arm"] == "EPS-human", spec.outcome].to_numpy()
    human = data.loc[data["arm"] == "Human", spec.outcome].to_numpy()
    estimate = float(np.mean(eps) - np.mean(human))
    eps_term = float(np.var(eps, ddof=1) / len(eps))
    human_term = float(np.var(human, ddof=1) / len(human))
    standard_error = math.sqrt(eps_term + human_term)
    degrees_of_freedom = (eps_term + human_term) ** 2 / (
        eps_term**2 / (len(eps) - 1) + human_term**2 / (len(human) - 1)
    )
    statistic = estimate / standard_error
    critical = float(stats.t.ppf(0.975, degrees_of_freedom))
    row = base_result_row(spec)
    row.update(
        {
            "model": "Reference: Welch unequal-variance t-test",
            "reporting_role": "Reference",
            "clustering": "None",
            "estimate": estimate,
            "standard_error": standard_error,
            "ci_95_lower": estimate - critical * standard_error,
            "ci_95_upper": estimate + critical * standard_error,
            "test_statistic": statistic,
            "reference_distribution": "Student t (Welch-Satterthwaite)",
            "degrees_of_freedom": degrees_of_freedom,
            "p_value": float(2 * stats.t.sf(abs(statistic), degrees_of_freedom)),
            "fixed_effect_se_method": "Welch unequal-variance standard error",
            "converged": True,
            "optimizer": "Closed-form Welch test",
        }
    )
    return row


def inverse_symmetric_square_root(matrix: np.ndarray) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    tolerance = (
        np.finfo(float).eps
        * max(matrix.shape)
        * max(1.0, float(np.max(eigenvalues)))
    )
    if np.any(eigenvalues <= tolerance):
        raise np.linalg.LinAlgError("CR2 residual-maker block is singular")
    return (eigenvectors * (1 / np.sqrt(eigenvalues))) @ eigenvectors.T


def cr2_row(
    data: pd.DataFrame,
    spec: CohortSpec,
    cluster_column: str,
    model: str,
    role: str,
) -> dict[str, Any]:
    cluster_count = int(data[cluster_column].nunique())
    if cluster_count < 3:
        raise ValueError(f"CR2 requires at least three clusters; found {cluster_count}")
    fitted = smf.ols(f"{spec.outcome} ~ arm_eps", data=data).fit()
    design = np.asarray(fitted.model.exog, dtype=float)
    residuals = np.asarray(fitted.resid, dtype=float)
    bread = np.linalg.inv(design.T @ design)
    arm_position = list(fitted.model.exog_names).index("arm_eps")
    contrast = np.zeros(design.shape[1])
    contrast[arm_position] = 1
    meat = np.zeros_like(bread)
    score_columns: list[np.ndarray] = []
    for indices in data.groupby(cluster_column, observed=True).indices.values():
        indices = np.asarray(indices, dtype=int)
        cluster_design = design[indices]
        adjustment = inverse_symmetric_square_root(
            np.eye(len(indices)) - cluster_design @ bread @ cluster_design.T
        )
        score = cluster_design.T @ adjustment @ residuals[indices]
        meat += np.outer(score, score)
        supported = adjustment @ cluster_design @ bread @ contrast
        projection = bread @ (cluster_design.T @ supported)
        residualized = -(design @ projection)
        residualized[indices] += supported
        score_columns.append(residualized)
    covariance = bread @ meat @ bread
    estimate = float(fitted.params.iloc[arm_position])
    standard_error = float(np.sqrt(covariance[arm_position, arm_position]))
    statistic = estimate / standard_error
    scores = np.column_stack(score_columns)
    gram = scores.T @ scores
    trace = float(np.sum(scores * scores))
    degrees_of_freedom = trace**2 / float(np.sum(gram * gram))
    critical = float(stats.t.ppf(0.975, degrees_of_freedom))
    row = base_result_row(spec)
    row.update(
        {
            "model": model,
            "reporting_role": role,
            "clustering": f"{cluster_column} ({cluster_count} clusters)",
            "estimate": estimate,
            "standard_error": standard_error,
            "ci_95_lower": estimate - critical * standard_error,
            "ci_95_upper": estimate + critical * standard_error,
            "test_statistic": statistic,
            "reference_distribution": "Student t (CR2 Satterthwaite)",
            "degrees_of_freedom": degrees_of_freedom,
            "p_value": float(2 * stats.t.sf(abs(statistic), degrees_of_freedom)),
            "fixed_effect_se_method": "Bell-McCaffrey CR2 cluster-robust covariance",
            "converged": True,
            "optimizer": "OLS with Bell-McCaffrey CR2 adjustment",
        }
    )
    return row


def mixed_model(
    data: pd.DataFrame, outcome: str, fixed_effect: str, cluster_column: str
) -> MixedFit:
    return fit_mixed_model(
        lambda: smf.mixedlm(
            f"{outcome} ~ {fixed_effect}",
            data=data,
            groups=data[cluster_column],
            re_formula="1",
        )
    )


def crossed_model(data: pd.DataFrame, outcome: str, fixed_effect: str) -> MixedFit:
    return fit_mixed_model(
        lambda: smf.mixedlm(
            f"{outcome} ~ {fixed_effect}",
            data=data,
            groups=data["all_observations"],
            re_formula="0",
            vc_formula={
                "manager": "0 + C(manager_id)",
                "community": "0 + C(community_id)",
            },
            use_sparse=True,
        )
    )


def component_map(result: Any) -> dict[str, float]:
    return {
        name: max(0.0, float(value))
        for name, value in zip(result.model.exog_vc.names, result.vcomp)
    }


def icc_row(
    spec: CohortSpec,
    fit: MixedFit,
    profile: ICCProfile,
    component: str,
    cluster_count: int,
    cluster_variance: float,
    other_variance: float = math.nan,
    role: str = "Primary",
    model: Optional[str] = None,
) -> dict[str, Any]:
    residual = float(fit.result.scale)
    return {
        "cohort": spec.name,
        "model": model or f"Unconditional {component} random-intercept model",
        "reporting_role": role,
        "component": component,
        "cluster_count": cluster_count,
        "cluster_variance": cluster_variance,
        "other_cluster_variance": other_variance,
        "residual_variance": residual,
        "icc": profile.point,
        "icc_95_lower": profile.lower,
        "icc_95_upper": profile.upper,
        "interval_method": profile.method,
        "interval_status": profile.status,
        "boundary_estimate": variance_on_boundary(cluster_variance, residual),
        "converged": bool(fit.result.converged),
        "optimizer": fit.method,
        "fit_attempts": json.dumps(fit.attempts, ensure_ascii=False),
    }


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_metadata(
    path: Path,
    data: pd.DataFrame,
    spec: CohortSpec,
    human_path: Path,
    eps_path: Path,
) -> None:
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": spec.name,
        "outcome": spec.outcome,
        "inputs": [
            {"path": str(human_path), "sha256": file_hash(human_path)},
            {"path": str(eps_path), "sha256": file_hash(eps_path)},
        ],
        "sample_size": int(len(data)),
        "sample_size_by_arm": data.groupby("arm").size().astype(int).to_dict(),
        "manager_count": int(data["manager_id"].nunique()),
        "community_count": int(data["community_id"].nunique()),
        "manager_count_by_arm": (
            data.groupby("arm")["manager_id"].nunique().astype(int).to_dict()
        ),
        "community_count_by_arm": (
            data.groupby("arm")["community_id"].nunique().astype(int).to_dict()
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "openpyxl": openpyxl.__version__,
        },
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def run_analysis(
    spec: CohortSpec, human_path: Path, eps_path: Path, outdir: Path
) -> None:
    human = read_arm(human_path, "Human", spec)
    eps = read_arm(eps_path, "EPS-human", spec)
    data = pd.concat([eps, human], ignore_index=True)
    outdir.mkdir(parents=True, exist_ok=True)
    validate_data(data, spec).to_csv(
        outdir / "data_checks.csv", index=False, encoding="utf-8-sig"
    )

    manager_summary = (
        data.groupby(["manager_id", "arm"], observed=True)
        .agg(n=(spec.outcome, "size"), mean_outcome=(spec.outcome, "mean"))
        .reset_index()
    )
    manager_summary["manager_sort"] = pd.to_numeric(
        manager_summary["manager_id"], errors="coerce"
    )
    manager_summary.sort_values(["manager_sort", "arm"]).drop(
        columns="manager_sort"
    ).to_csv(outdir / "manager_summary_by_arm.csv", index=False, encoding="utf-8-sig")
    (
        data.groupby(["community_id", "arm"], observed=True)
        .agg(
            n=(spec.outcome, "size"),
            represented_managers=("manager_id", "nunique"),
            mean_outcome=(spec.outcome, "mean"),
        )
        .reset_index()
        .sort_values(["arm", "community_id"])
        .to_csv(outdir / "community_summary.csv", index=False, encoding="utf-8-sig")
    )

    null_manager = mixed_model(data, spec.outcome, "1", "manager_id")
    manager_variance = random_intercept_variance(null_manager.result)
    manager_point = manager_variance / (manager_variance + null_manager.result.scale)
    manager_profile = (
        unavailable_profile(manager_point, "unavailable_boundary_estimate")
        if variance_on_boundary(manager_variance, float(null_manager.result.scale))
        else profile_icc_ci(data, "manager_id", spec.outcome)
    )
    arm_manager = mixed_model(data, spec.outcome, "arm_eps", "manager_id")
    arm_manager_variance = random_intercept_variance(arm_manager.result)

    null_community = mixed_model(data, spec.outcome, "1", "community_id")
    community_variance = random_intercept_variance(null_community.result)
    community_point = community_variance / (
        community_variance + null_community.result.scale
    )
    community_profile = (
        profile_icc_ci(data, "community_id", spec.outcome)
        if spec.profile_community_icc
        else unavailable_profile(community_point, "not_calculated")
    )
    arm_community = mixed_model(data, spec.outcome, "arm_eps", "community_id")
    arm_community_variance = random_intercept_variance(arm_community.result)

    model_rows = [
        welch_row(data, spec),
        extract_arm_effect(
            arm_manager,
            spec,
            "Primary LMM: manager random intercept",
            "Primary",
            f"manager_id ({data['manager_id'].nunique()} managers)",
            arm_manager_variance,
            None,
        ),
        extract_arm_effect(
            arm_community,
            spec,
            "Sensitivity LMM: community random intercept",
            "Sensitivity",
            f"community_id ({data['community_id'].nunique()} communities)",
            None,
            arm_community_variance,
        ),
        cr2_row(
            data,
            spec,
            "manager_id",
            "Sensitivity OLS: manager-clustered CR2 standard errors",
            "Sensitivity",
        ),
        cr2_row(
            data,
            spec,
            "community_id",
            "Supplementary OLS: community-clustered CR2 standard errors",
            "Supplementary",
        ),
    ]

    crossed_data = data.copy()
    crossed_data["all_observations"] = "all"
    null_crossed = crossed_model(crossed_data, spec.outcome, "1")
    null_components = component_map(null_crossed.result)
    null_total = sum(null_components.values()) + float(null_crossed.result.scale)
    arm_crossed = crossed_model(crossed_data, spec.outcome, "arm_eps")
    arm_components = component_map(arm_crossed.result)
    model_rows.append(
        extract_arm_effect(
            arm_crossed,
            spec,
            "Supplementary LMM: crossed manager and community random intercepts",
            "Supplementary",
            "manager_id + community_id (crossed)",
            arm_components["manager"],
            arm_components["community"],
        )
    )
    pd.DataFrame(model_rows).to_csv(
        outdir / "model_results.csv", index=False, encoding="utf-8-sig"
    )

    crossed_model_name = "Unconditional crossed manager-community model"
    crossed_profile = unavailable_profile(math.nan, "not_calculated")
    icc_rows = [
        icc_row(
            spec,
            null_manager,
            manager_profile,
            "manager",
            int(data["manager_id"].nunique()),
            manager_variance,
        ),
        icc_row(
            spec,
            null_community,
            community_profile,
            "community",
            int(data["community_id"].nunique()),
            community_variance,
            role="Sensitivity",
        ),
        icc_row(
            spec,
            null_crossed,
            ICCProfile(
                null_components["manager"] / null_total,
                crossed_profile.lower,
                crossed_profile.upper,
                crossed_profile.method,
                crossed_profile.status,
            ),
            "manager",
            int(data["manager_id"].nunique()),
            null_components["manager"],
            null_components["community"],
            "Supplementary",
            crossed_model_name,
        ),
        icc_row(
            spec,
            null_crossed,
            ICCProfile(
                null_components["community"] / null_total,
                crossed_profile.lower,
                crossed_profile.upper,
                crossed_profile.method,
                crossed_profile.status,
            ),
            "community",
            int(data["community_id"].nunique()),
            null_components["community"],
            null_components["manager"],
            "Supplementary",
            crossed_model_name,
        ),
    ]
    pd.DataFrame(icc_rows).to_csv(
        outdir / "icc_results.csv", index=False, encoding="utf-8-sig"
    )
    write_metadata(outdir / "analysis_metadata.json", data, spec, human_path, eps_path)


def main() -> None:
    args = parse_args()
    run_analysis(
        SPECS[args.cohort],
        args.human_xlsx.resolve(),
        args.eps_xlsx.resolve(),
        args.outdir.resolve(),
    )
    print(f"Outputs written to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
