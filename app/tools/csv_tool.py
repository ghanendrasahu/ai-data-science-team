"""CSV loading, saving and automatic dataset profiling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV with encoding fallback so real-world files keep working."""
    path = Path(path)
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
        except pd.errors.ParserError:
            return pd.read_csv(path, encoding=encoding, engine="python")
    raise ValueError(f"Could not read CSV file: {path}")


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _classify_dtype(dtype: np.dtype) -> str:
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    return "categorical"


def infer_column_types(df: pd.DataFrame) -> dict[str, str]:
    """Map every column to a coarse type used across the pipeline."""
    return {col: _classify_dtype(df[col].dtype) for col in df.columns}


def profile(df: pd.DataFrame) -> dict[str, Any]:
    """Produce a compact, JSON-serialisable profile of a dataframe.

    Used by the planner and chat agents so the LLM sees numbers, not raw data.
    """
    profile: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_types": infer_column_types(df),
        "missing": {},
        "duplicates": int(df.duplicated().sum()),
        "numeric_summary": {},
        "cardinality": {},
    }

    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        profile["missing"][col] = missing
        profile["cardinality"][col] = int(series.nunique())

        if _classify_dtype(series.dtype) == "numeric":
            profile["numeric_summary"][col] = {
                key: _round(value)
                for key, value in series.describe().to_dict().items()
            }

    target_hint = _suggest_target(df)
    profile["suggested_target"] = target_hint
    return profile


def _suggest_target(df: pd.DataFrame) -> str | None:
    """Heuristic: prefer a low-cardinality categorical column as the target.

    Falls back to the lowest-cardinality numeric column (so all-numeric
    datasets still get a usable supervised target), then to the last column.
    """
    if df.empty or len(df.columns) == 0:
        return None

    categoricals = [
        (col, int(df[col].nunique()))
        for col in df.columns
        if _classify_dtype(df[col].dtype) == "categorical"
    ]
    if categoricals:
        return min(categoricals, key=lambda item: (item[1], item[0]))[0]

    numerics = [
        (col, int(df[col].nunique()))
        for col in df.columns
        if _classify_dtype(df[col].dtype) == "numeric"
    ]
    if numerics:
        return min(numerics, key=lambda item: (item[1], item[0]))[0]

    return df.columns[-1]


def _round(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return round(float(value), 4)
    if isinstance(value, np.integer):
        return int(value)
    return value