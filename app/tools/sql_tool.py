"""Read-only SQL querying of datasets loaded from CSV.

Datasets are loaded into an in-memory SQLite database (per call) so agents can
answer questions with real queries while staying fully read-only. Only SELECT
statements are allowed; any other statement is rejected before execution.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from app.tools.csv_tool import load_csv

_ALLOWED = re.compile(r"^\s*SELECT\b", re.IGNORECASE)


def query_dataset(
    dataset_path: str,
    sql: str,
    table_name: str = "data",
    limit: int = 200,
) -> dict[str, Any]:
    """Run a read-only SELECT against the given CSV and return rows + columns.

    Returns a dict with ``columns`` and ``rows`` (list of lists), trimmed to
    ``limit`` rows so large results never blow up the LLM context.
    """
    if not _ALLOWED.match(sql):
        raise ValueError("Only SELECT queries are allowed.")

    df = load_csv(dataset_path)
    conn = sqlite3.connect(":memory:")
    try:
        df.to_sql(table_name, conn, index=False, if_exists="replace")
        cursor = conn.execute(sql)
        columns = [description[0] for description in cursor.description]
        rows = [list(map(_serialise, row)) for row in cursor.fetchmany(limit)]
        return {"columns": columns, "rows": rows, "truncated": len(rows) == limit}
    finally:
        conn.close()


def _serialise(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


__all__ = ["query_dataset"]
