"""Lightweight time-series forecasting (statsmodels-backed)."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from app.tools.csv_tool import load_csv


def forecast_series(
    values: pd.Series,
    horizon: int = 10,
    period: int | None = None,
) -> dict[str, Any]:
    """Forecast the next ``horizon`` steps of a numeric series.

    Method ladder, chosen by data size:
    - Holt-Winters (trend + seasonality) when ``period`` is given and the
      series covers at least two seasons;
    - Holt-style trend smoothing otherwise;
    - flat naive forecast as the last-resort fallback.
    """
    if horizon < 1 or horizon > 90:
        raise ValueError("horizon must be between 1 and 90.")
    series = pd.to_numeric(pd.Series(values).reset_index(drop=True), errors="coerce").dropna()
    if len(series) < 8:
        raise ValueError("Forecasting needs at least 8 numeric points.")

    fitted = None
    method = "naive"
    try:  # statsmodels is imported lazily; the tool stays usable without it
        if period and period > 1 and len(series) >= 2 * period:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = ExponentialSmoothing(
                    series.astype(float),
                    trend="add",
                    seasonal="add",
                    seasonal_periods=period,
                ).fit(optimized=True)
            method = f"holt_winters(period={period})"
        else:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = ExponentialSmoothing(series.astype(float), trend="add").fit(optimized=True)
            method = "holt_trend"
    except Exception:  # noqa: BLE001 - forecasting must never crash the run
        fitted = None
        method = "naive"

    if fitted is not None:
        predicted = np.asarray(fitted.forecast(horizon), dtype=float)
    else:
        predicted = np.full(horizon, float(series.tail(min(len(series), 10)).mean()))

    predicted = np.nan_to_num(predicted, nan=float(series.median()))
    return {
        "method": method,
        "history_points": int(len(series)),
        "last_value": round(float(series.iloc[-1]), 4),
        "forecast": [
            {"step": i + 1, "value": round(float(v), 4)}
            for i, v in enumerate(predicted)
        ],
    }


def forecast_csv(
    dataset_path: str,
    column: str | None = None,
    horizon: int = 10,
    period: int | None = None,
) -> dict[str, Any]:
    """Forecast a CSV column: explicit ``column`` or the first numeric one."""
    df = load_csv(dataset_path)
    if column:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found; available: {list(df.columns)}")
        target_col = column
    else:
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            raise ValueError("No numeric column available for forecasting - name one.")
        target_col = numeric.columns[0]

    result: dict[str, Any] = forecast_series(df[target_col], horizon=horizon, period=period)
    result["column"] = target_col
    return result


__all__ = ["forecast_csv", "forecast_series"]
