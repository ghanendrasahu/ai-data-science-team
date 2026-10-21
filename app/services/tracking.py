"""Experiment tracking and run monitoring.

A lightweight, dependency-free tracker writes keys/metrics per run into the same
SQLite store used by the app. If MLflow is installed and configured it also syncs
to an MLflow tracking server, but the core API works without it.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from typing import Any

try:
    import mlflow  # type: ignore[import-not-found]

    _MLFLOW_AVAILABLE = True
except Exception:  # noqa: BLE001 - mlflow is optional
    mlflow = None  # type: ignore[assignment]
    _MLFLOW_AVAILABLE = False


logger = logging.getLogger(__name__)

_LOCK = threading.Lock()


def available() -> bool:
    """Return True when an MLflow client is importable."""
    return _MLFLOW_AVAILABLE


def log_run(
    run_id: str,
    params: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    artifact_dir: str | None = None,
) -> dict[str, Any]:
    """Record a step / final result for a run.

    Always persists to the local store; when MLflow is enabled it mirrors the
    same values to the configured tracking server.
    """
    snapshot = _persist_snapshot(run_id, params=params, metrics=metrics, tags=tags)
    if not available():
        return snapshot
    _mlflow_sync(run_id, params=params, metrics=metrics, tags=tags, artifact_dir=artifact_dir)
    return snapshot


def recent_runs(limit: int = 10) -> list[dict[str, Any]]:
    """Return recent run summaries for dashboards / monitoring."""
    from app.services.store import get_store

    records = get_store().list_runs(limit)
    out = []
    for record in records:
        full = get_store().get_run(record["run_id"]) or record
        run_id = record["run_id"]
        metrics = _load_metrics(run_id)
        out.append(
            {
                "run_id": run_id,
                "status": record["status"],
                "created_at": record["created_at"],
                "best_model": _best_model_name(full),
                "metrics": metrics,
            }
        )
    return out


def _persist_snapshot(
    run_id: str,
    params: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    tags: dict[str, str] | None,
) -> dict[str, Any]:
    import sqlite3

    from app.config import get_settings

    db = str(get_settings().sqlite_path)
    with _LOCK, sqlite3.connect(db) as conn:
        conn.execute(_SCHEMA)
        conn.execute(
            "INSERT INTO run_metrics (run_id, params, metrics, tags, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id) DO UPDATE SET params=excluded.params, "
            "metrics=excluded.metrics, tags=excluded.tags, updated_at=excluded.updated_at",
            (
                run_id,
                json.dumps(params or {}),
                json.dumps(metrics or {}),
                json.dumps(tags or {}),
                datetime.now(UTC).isoformat(),
            ),
        )
    return {"run_id": run_id, "params": params or {}, "metrics": metrics or {}}


_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_metrics (
    run_id TEXT PRIMARY KEY,
    params TEXT,
    metrics TEXT,
    tags TEXT,
    updated_at TEXT NOT NULL
);
"""


def _load_metrics(run_id: str) -> dict[str, Any]:
    import sqlite3

    from app.config import get_settings

    db = str(get_settings().sqlite_path)
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT metrics, params FROM run_metrics WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        return {}
    metrics = json.loads(row["metrics"] or "{}")
    return metrics


def _best_model_name(record: dict[str, Any]) -> str | None:
    result = record.get("result") or {}
    best = result.get("best_model") or {}
    return best.get("model") if isinstance(best, dict) else None


def _mlflow_sync(
    run_id: str,
    params: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    tags: dict[str, str] | None,
    artifact_dir: str | None,
) -> None:
    try:
        with mlflow.start_run(run_id=run_id, run_name=run_id):  # type: ignore[union-attr]
            for key, value in (params or {}).items():
                mlflow.log_param(key, value)  # type: ignore[union-attr]
            for key, value in (metrics or {}).items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)  # type: ignore[union-attr]
            for key, value in (tags or {}).items():
                mlflow.set_tag(key, value)  # type: ignore[union-attr]
            if artifact_dir:
                mlflow.log_artifacts(artifact_dir)  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001 - tracking must never break the pipeline
        logger.warning("MLflow sync failed: %s", exc)


__all__ = ["available", "log_run", "recent_runs"]