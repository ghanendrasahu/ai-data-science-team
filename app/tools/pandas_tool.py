"""Pandas-level data preparation helpers shared by the cleaning and
feature engineering agents."""

from __future__ import annotations

import pandas as pd

NUMERIC_TYPES = ("int16", "int32", "int64", "float16", "float32", "float64")


def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def is_categorical(series: pd.Series) -> bool:
    return not is_numeric(series) and not pd.api.types.is_datetime64_any_dtype(series)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values: median for numerics, mode for categoricals."""
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        if is_numeric(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "unknown")
    return df


def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that carry no information (zero variance).

    Falls back to the original frame if every column is constant so the
    downstream agents never receive an empty dataframe.
    """
    keep = [col for col in df.columns if df[col].nunique() > 1]
    return df[keep] if keep else df


def winsorize_outliers(df: pd.DataFrame, limits: tuple[float, float] = (0.01, 0.99)) -> pd.DataFrame:
    """Cap numeric outliers at the given quantile limits (winsorisation).

    Uses ``clip`` between the requested quantiles instead of ``scipy``'s
    ``winsorize``, which can collapse small or degenerate samples down to a
    single constant value.
    """
    lower, upper = limits
    for col in df.columns:
        if not is_numeric(df[col]):
            continue
        series = df[col]
        lo = series.quantile(lower)
        hi = series.quantile(upper)
        if pd.isna(lo) or pd.isna(hi) or lo == hi:
            continue
        df[col] = series.clip(lower=lo, upper=hi)
    return df


def prepare_for_ml(
    df: pd.DataFrame, target: str, drop_cols: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Split into features + target and one-hot encode categorical features.

    Returns ``(X, y, feature_names)`` ready for scikit-learn estimators.
    """
    df = df.copy()
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")

    y = df[target]
    X = df.drop(columns=[target])

    drop_in_target = y.isna()
    if drop_in_target.any():
        X = X[~drop_in_target]
        y = y[~drop_in_target]

    encoded_parts: list[pd.DataFrame] = []
    numeric_cols: list[str] = []
    for col in X.columns:
        if is_numeric(X[col]):
            numeric_cols.append(col)
            continue
        if is_categorical(X[col]) and X[col].nunique() <= 50:
            encoded_parts.append(pd.get_dummies(X[col], prefix=col, drop_first=False))
        else:
            X[col] = X[col].astype("category").cat.codes
            numeric_cols.append(col)

    feature_names = list(numeric_cols)
    if encoded_parts:
        encoded = pd.concat(encoded_parts, axis=1)
        feature_names.extend(encoded.columns.tolist())
        X = pd.concat([X[numeric_cols].reset_index(drop=True), encoded.reset_index(drop=True)], axis=1)
    else:
        X = X[numeric_cols]

    return X.reset_index(drop=True), y.reset_index(drop=True), feature_names