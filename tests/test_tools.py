"""Unit tests for the tool layer (no LLM needed)."""

import sqlite3
from pathlib import Path

import pytest

from app.tools.sql_tool import query_dataset


@pytest.fixture()
def csv_file(tmp_path: Path) -> str:
    path = tmp_path / "sales.csv"
    path.write_text("region,sales\nEU,100\nUS,250\nEU,60\n", encoding="utf-8")
    return str(path)


def test_query_dataset_select_aggregation(csv_file: str):
    result = query_dataset(csv_file, "SELECT region, SUM(sales) AS total FROM data GROUP BY region")
    by_region = {
        row[result["columns"].index("region")]: row[result["columns"].index("total")]
        for row in result["rows"]
    }
    assert by_region == {"EU": 160, "US": 250}
    assert set(result["columns"]) == {"region", "total"}


def test_query_dataset_rejects_non_select(csv_file: str):
    with pytest.raises(ValueError, match="SELECT"):
        query_dataset(csv_file, "DELETE FROM data")


def test_query_dataset_truncates_to_limit(csv_file: str):
    # Pure SELECT (the tool rejects CTEs): cross join 3x3x3 = 27 rows.
    long_sql = "SELECT a.region FROM data a, data b, data c"
    result = query_dataset(csv_file, long_sql, limit=10)
    assert len(result["rows"]) == 10
    assert result["truncated"] is True


def test_query_dataset_serialises_sqlite_error(csv_file: str):
    with pytest.raises(sqlite3.OperationalError):
        query_dataset(csv_file, "SELECT * FROM nonexistent_table")


def test_query_dataset_no_side_effects(tmp_path: Path):
    """The SQLite db is in-memory; nothing should leave a file behind."""
    marker = tmp_path / "marker"
    marker.write_text("x", encoding="utf-8")
    marker_bytes = marker.read_bytes()
    csv_file = tmp_path / "s.csv"
    csv_file.write_text("a\n1\n2\n", encoding="utf-8")
    query_dataset(str(csv_file), "SELECT COUNT(*) AS n FROM data")
    assert marker.read_bytes() == marker_bytes
    assert {p.name for p in tmp_path.iterdir()} == {"marker", "s.csv"}
