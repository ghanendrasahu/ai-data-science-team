"""Security tests: CORS allowlist, API-key gate and path containment.

Covers the audit's cross-cutting findings:
- ``allow_origins=["*"]`` + ``allow_credentials=True`` (browser-rejected combo)
- unauthenticated mutating endpoints
- arbitrary file read via ``/sql``'s ``dataset_path``
"""

from fastapi.testclient import TestClient

from app import main


def _client() -> TestClient:
    with TestClient(main.app) as client:
        return client


def test_cors_attacker_origin_gets_no_allow_header():
    client = _client()
    response = client.options(
        "/health",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers.get("access-control-allow-origin") is None


def test_cors_allowed_origin_is_echoed():
    client = _client()
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_write_route_open_without_api_key():
    client = _client()
    response = client.post("/chat", json={"question": "what do you see?"})
    assert response.status_code == 200


def test_write_routes_reject_wrong_key(monkeypatch):
    monkeypatch.setattr(main.settings, "api_key", "secret-123")
    client = _client()
    response = client.post("/chat", json={"question": "hello"})
    assert response.status_code == 401

    ok = client.post(
        "/chat",
        json={"question": "hello"},
        headers={"X-API-Key": "secret-123"},
    )
    assert ok.status_code == 200


def test_sql_rejects_dataset_path_outside_uploads(tmp_path):
    client = _client()
    outside = tmp_path / "secret.csv"
    outside.write_text("a,b\n1,2\n", encoding="utf-8")
    response = client.post("/sql", json={"question": "rows?", "dataset_path": str(outside)})
    assert response.status_code == 400
    assert "uploaded dataset" in response.json()["detail"]


def test_sql_flattens_traversal_names():
    """A relative `../x.csv` must be flattened to a file *inside* uploads/."""
    response = _client().post("/sql", json={"question": "rows?", "dataset_path": "../../x.csv"})
    # Resolves inside uploads/ -> simply "not found", never an escape.
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]
