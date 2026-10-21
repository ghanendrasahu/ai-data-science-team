"""Anomaly detection tool - IsolationForest over numeric columns."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.05,
    max_rows: int = 25,
) -> dict[str, Any]:
    """Flag the most anomalous rows in ``df`` and explain the top driver.

    Requires at least two non-constant numeric columns; returns the flagged
    row indices, their scores and, per row, the feature contributing most to
    the anomaly (largest absolute z-score).
    """
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        raise ValueError("Anomaly detection requires at least 2 numeric columns.")

    numeric = numeric.fillna(numeric.median(numeric_only=True))
    keep = numeric.columns[numeric.nunique() > 1].tolist()
    if len(keep) < 2:
        raise ValueError("Anomaly detection requires at least 2 non-constant numeric columns.")
    numeric = numeric[keep]

    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    scaled = StandardScaler().fit_transform(numeric)
    detector = IsolationForest(contamination=max(contamination, 0.001), random_state=42)
    detector.fit(scaled)
    scores = -detector.score_samples(scaled)
    z_scores = np.abs(scaled)

    flagged_indices = np.argsort(scores)[::-1]
    threshold = float(np.quantile(scores, 1 - max(contamination, 1 / len(scores))))
    flagged = [int(i) for i in flagged_indices if scores[i] >= threshold][:max_rows]

    rows = []
    for i in flagged:
        top_feature_idx = int(np.argmax(z_scores[i]))
        rows.append(
            {
                "row_index": int(df.index[int(i)]),
                "anomaly_score": round(float(scores[i]), 4),
                "top_feature": keep[top_feature_idx],
                "top_z": round(float(z_scores[i][top_feature_idx]), 3),
            }
        )

    return {
        "method": "IsolationForest",
        "n_rows": int(len(df)),
        "n_flagged": int(sum(scores >= threshold)),
        "rows": rows,
        "features_used": keep,
        "threshold_score": round(threshold, 4),
    }


def summarise_anomalies(result: dict[str, Any]) -> str:
    """One-paragraph, report-friendly summary of a detection result."""
    rows = result.get("rows", [])
    top = ", ".join(r["top_feature"] for r in rows[:3]) or "n/a"
    return json.dumps(
        {
            "flagged": result.get("n_flagged", 0),
            "of": result.get("n_rows"),
            "worst_drivers": top,
            "threshold": result.get("threshold_score"),
        }
    )


__all__ = ["detect_anomalies", "summarise_anomalies"]
