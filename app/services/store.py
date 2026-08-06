"""SQLite-backed persistence for workflow runs.

Each run stores its id, status, input path and the final agent state as JSON,
so clients can poll for progress and retrieve results later.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from app.config import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    dataset_path TEXT,
    target TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    result_json TEXT,
    error TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    """Thread-safe wrapper around a single SQLite connection per thread."""

    def __init__(self, db_path=None) -> None:
        self.db_path = str(db_path or get_settings().sqlite_path)
        self._lock = threading.Lock()
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(_SCHEMA)

    def create_run(self, dataset_path: str, target: str | None = None) -> str:
        run_id = uuid.uuid4().hex[:12]
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO runs (run_id, status, dataset_path, target, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, "pending", dataset_path, target, _now()),
            )
        return run_id

    def set_status(self, run_id: str, status: str, error: str | None = None) -> None:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE runs SET status = ?, error = ?, finished_at = ? WHERE run_id = ?",
                (status, error, _now() if status in ("succeeded", "failed") else None, run_id),
            )

    def set_result(self, run_id: str, result: dict[str, Any]) -> None:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE runs SET result_json = ?, status = 'succeeded', finished_at = ? "
                "WHERE run_id = ?",
                (json.dumps(result, default=str), _now(), run_id),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["result"] = json.loads(record.pop("result_json")) if record.get("result_json") else None
        return record

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT run_id, status, dataset_path, target, created_at, finished_at "
                "FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(zip(("run_id", "status", "dataset_path", "target", "created_at", "finished_at"), row)) for row in rows]


_run_store: RunStore | None = None


def get_store() -> RunStore:
    global _run_store
    if _run_store is None:
        _run_store = RunStore()
    return _run_store