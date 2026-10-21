"""Endpoint smoke tests + one real end-to-end workflow run (mock LLM, no keys)."""

import io

import pandas as pd
from fastapi.testclient import TestClient

from app import main


def _csv_bytes(n: int = 60) -> bytes:
    frame = pd.DataFrame(
        {
            "x1": range(n),
            "x2": [i % 7 for i in range(n)],
            "y": ["yes" if i % 3 == 0 else "no" for i in range(n)],
        }
    )
    return frame.to_csv(index=False).encode("utf-8")


def test_root_metadata():
    client = TestClient(main.app)
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["agents"] and isinstance(body["agents"], list)
    assert "provider" in body


def test_health():
    client = TestClient(main.app)
    assert client.get("/health").status_code == 200


def test_upload_rejects_non_csv():
    client = TestClient(main.app)
    response = client.post(
        "/upload",
        files={"file": ("evil.exe", io.BytesIO(b"MZ..."), "application/octet-stream")},
    )
    assert response.status_code == 400


def test_workflow_missing_dataset():
    client = TestClient(main.app)
    response = client.post("/workflow", json={"filename": "nope.csv", "target": "y"})
    assert response.status_code == 404


def test_full_workflow_end_to_end():
    """The repo's core promise: upload -> autonomous run -> persisted report."""
    with TestClient(main.app) as live:
        upload = live.post("/upload", files={"file": ("tiny.csv", io.BytesIO(_csv_bytes()), "text/csv")})
        assert upload.status_code == 200, upload.text

        started = live.post("/workflow", json={"filename": "tiny.csv", "target": "y", "tune": False})
        assert started.status_code == 200, started.text
        run_id = started.json()["run_id"]

        record = live.get(f"/workflow/{run_id}").json()
        assert record["status"] == "succeeded", record.get("error")

        reports = live.get("/reports").json()
        assert any(r["name"] == f"report_{run_id}.md" for r in reports)

        runs = live.get("/runs").json()
        assert any(r["run_id"] == run_id for r in runs)
