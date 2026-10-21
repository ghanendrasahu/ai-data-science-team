"""Statistical hypothesis testing tool (scipy-backed, deterministic)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SIG_ALPHA = 0.05


def run_statistical_test(df: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a hypothesis test and return its statistic + p-value.

    Supported specs:
      ``ttest``      two independent numeric groups (t-test)
      ``chi2``       association between two categorical columns
      ``correlation`` pearson/spearman between two numeric columns
      ``normality``  shapiro wruble normality of one numeric column
      ``anova``      one numeric column across 2+ groups
    """
    test = str(spec.get("test", "")).lower()
    if test == "ttest":
        return _ttest(df, spec)
    if test == "chi2":
        return _chi2(df, spec)
    if test in {"correlation", "corr"}:
        return _correlation(df, spec)
    if test == "normality":
        return _normality(df, spec)
    if test == "anova":
        return _anova(df, spec)
    raise ValueError(
        f"Unknown test '{test}'. Supported: ttest, chi2, correlation, normality, anova."
    )


def _ttest(df: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
    from scipy.stats import ttest_ind

    value_col, group_col = spec["column"], spec["group_column"]
    group_a = df.loc[df[group_col] == spec["group_a"], value_col].dropna()
    group_b = df.loc[df[group_col] == spec["group_b"], value_col].dropna()
    if len(group_a) < 2 or len(group_b) < 2:
        raise ValueError("Both groups need at least 2 rows.")
    stat, p = ttest_ind(group_a, group_b, equal_var=spec.get("equal_var", False))
    return _result("ttest", stat, p, group_a=spec["group_a"], group_b=spec["group_b"])


def _chi2(df: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
    from scipy.stats import chi2_contingency

    contingency = pd.crosstab(df[spec["column_a"]], df[spec["column_b"]])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        raise ValueError("Chi-square needs at least a 2x2 table.")
    stat, p, dof, _ = chi2_contingency(contingency)
    return _result("chi2", stat, p, dof=int(dof))


def _correlation(df: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
    method = spec.get("method", "pearson").lower()
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'.")
    pair = df[[spec["column_a"], spec["column_b"]]].dropna()
    if len(pair) < 3:
        raise ValueError("Correlation needs at least 3 complete rows.")
    corr = pair.corr(method=method).iloc[0, 1]
    return _result(method, corr, None, note="correlation coefficient (|r|>0.5 is strong)")


def _normality(df: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
    from scipy.stats import shapiro

    values = df[spec["column"]].dropna()
    if len(values) < 3:
        raise ValueError("Normality test needs at least 3 values.")
    if len(values) > 5000:
        values = values.sample(5000, random_state=42)
    stat, p = shapiro(values)
    return _result("normality", stat, p, note="H0: data is normal")


def _anova(df: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
    from scipy.stats import f_oneway

    groups = df.groupby(spec["group_column"])[spec["column"]].apply(lambda s: s.dropna().tolist())
    groups = [np.asarray(v) for v in groups.tolist() if len(v) >= 2]
    if len(groups) < 2:
        raise ValueError("ANOVA needs at least 2 groups with 2+ rows.")
    stat, p = f_oneway(*groups)
    return _result("anova", stat, p, n_groups=len(groups))


def _result(
    test: str,
    statistic: float | None,
    p_value: float | None,
    **extra: Any,
) -> dict[str, Any]:
    interpretation = (
        "n/a for correlation"
        if p_value is None
        else (
            f"statistically significant at alpha={SIG_ALPHA}"
            if float(p_value) < SIG_ALPHA
            else "not statistically significant"
        )
    )
    return {
        "test": test,
        "statistic": round(float(statistic), 6) if statistic is not None else None,
        "p_value": round(float(p_value), 6) if p_value is not None and np.isfinite(p_value) else None,
        "significant": None if p_value is None else bool(float(p_value) < SIG_ALPHA),
        "interpretation": interpretation,
        **extra,
    }


__all__ = ["run_statistical_test"]
