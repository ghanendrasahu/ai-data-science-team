"""Tests for the Phase-B tool layer: sandboxed python, anomalies, stats,
forecast - both as tools and through the API (auth + containment included)."""

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import main
from app.tools.anomaly_tool import detect_anomalies
from app.tools.forecast_tool import forecast_series
from app.tools.stats_tool import run_statistical_test


# --------------------------------------------------------------------------- #
# Tool units
# --------------------------------------------------------------------------- #
def _df(n: int = 60) -> pd.DataFrame:
    return pd.DataFrame({"x": range(n), "y": [i * 2.0 for i in range(n)]})


def test_anomalies_flags_injected_outlier():
    df = _df(40)
    df.loc[len(df)] = {"x": 1_000, "y": -9_000.0}
    result = detect_anomalies(df, contamination=0.05)
    assert result["method"] == "IsolationForest"
    assert result["n_flagged"] >= 1
    assert result["rows"][0]["row_index"] == len(df) - 1
    assert result["rows"][0]["top_feature"] in {"x", "y"}


def test_anomalies_requires_numeric_columns():
    df = pd.DataFrame({"a": ["x", "y"], "b": ["1", "2"]})
    with pytest.raises(ValueError, match="numeric"):
        detect_anomalies(df)


def test_stats_ttest_detects_group_difference():
    df = pd.DataFrame(
        {
            "value": [100.0, 102.0, 98.0, 101.0, 99.0, 140.0, 141.0, 139.0, 142.0, 140.5],
            "group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
        }
    )
    result = run_statistical_test(
        df,
        {"test": "ttest", "column": "value", "group_column": "group", "group_a": "A", "group_b": "B"},
    )
    assert result["significant"] is True


def test_stats_unknown_test_rejected():
    with pytest.raises(ValueError, match="Unknown test"):
        run_statistical_test(_df(10), {"test": "quantum"})


def test_forecast_returns_horizon_values():
    series = pd.Series([float(10 + i * 2 + (i % 7)) for i in range(40)])
    result = forecast_series(series, horizon=5, period=7)
    assert len(result["forecast"]) == 5
    assert all(f["value"] == f["value"] for f in result["forecast"])  # no NaNs
    assert result["method"] in {"holt_trend", "holt_winters(period=7)", "naive"}


# --------------------------------------------------------------------------- #
# API layer (gated + contained)
# --------------------------------------------------------------------------- #
def _client() -> TestClient:
    with TestClient(main.app) as live:
        return live


def _upload_csv(live: TestClient, name: str, frame: pd.DataFrame) -> None:
    upload = live.post(
        "/upload",
        files={"file": (name, io.BytesIO(frame.to_csv(index=False).encode()), "text/csv")},
    )
    assert upload.status_code == 200, upload.text


def test_api_python_executes_and_captures():
    live = _client()
    _upload_csv(live, "pydata.csv", _df(12))
    response = live.post(
        "/python",
        json={
            "code": "import pandas as pd\ndf = pd.read_csv('pydata.csv')\nprint(round(df['x'].sum()))",
            "data_files": ["pydata.csv"],
        },
    )
    body = response.json()
    assert response.status_code == 200, body
    assert body["returncode"] == 0
    assert body["stdout"].endswith("66")  # 0+1+...+11


def test_api_python_rejects_denied_import():
    live = _client()
    response = live.post("/python", json={"code": "import socket\nprint(socket)"})
    assert response.status_code == 400
    assert "denied" in response.json()["detail"]


def test_api_python_timeout_kills_snippet():
    live = _client()
    response = live.post(
        "/python",
        json={"code": "import time\ntime.sleep(30)", "timeout_seconds": 2},
    )
    body = response.json()
    assert response.status_code == 200
    assert body["timed_out"] is True
    assert body["returncode"] == 124


def test_api_anomalies_end_to_end():
    frame = _df(50)
    frame.loc[len(frame)] = {"x": 9999.0, "y": -9999.0}
    live = _client()
    _upload_csv(live, "anom.csv", frame)
    response = live.post("/anomalies", json={"dataset_path": "anom.csv"})
    assert response.status_code == 200
    assert response.json()["n_flagged"] >= 1


def test_api_stats_and_forecast_gated_by_key(monkeypatch):
    monkeypatch.setattr(main.settings, "api_key", "secret-key")
    live = _client()

    blocked = live.post("/stats", json={"dataset_path": "x.csv", "spec": {"test": "ttest"}})
    assert blocked.status_code == 401

    ok = live.post("/forecast", json={"dataset_path": "x.csv"}, headers={"X-API-Key": "secret-key"})
    # Auth passes; then the (missing) dataset path produces 404
    assert ok.status_code == 404


def test_api_tool_containment(monkeypatch):
    """Attacker path outside uploads must never be read by these tools."""
    live = _client()
    for route, payload in [
        ("/anomalies", {"dataset_path": "C:/Windows/win.ini"}),
        ("/forecast", {"dataset_path": "../../.env"}),
        ("/stats", {"dataset_path": "/etc/passwd", "spec": {"test": "ttest"}}),
    ]:
        response = live.post(route, json=payload)
        assert response.status_code in {400, 404}
