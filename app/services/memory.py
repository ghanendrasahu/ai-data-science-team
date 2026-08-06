"""Lightweight cross-run memory for context retrieval.

Implements a small vector memory over run summaries. Texts are embedded with a
TF-IDF vectoriser (no external vector DB or heavy ML dependency) and stored in
SQLite; retrieval returns the most relevant prior run summaries by cosine
similarity so agents can remember what has been learned before.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings

_LOCK = threading.Lock()


def embed_store(text: str, metadata: dict[str, Any] | None = None, key: str = "") -> str:
    """Store a text snippet with its embedding and return a document id."""
    record_id = key or uuid.uuid4().hex[:12]
    vector = _vector([text])[0]
    _upsert(record_id, text, vector, metadata)
    return record_id


def search(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Return the top-k most similar stored documents to ``query``."""
    docs = _documents()
    if not docs:
        return []
    query_vec = _vector([query])[0]
    matrix = np.vstack([doc["vector"] for doc in docs])
    sims = _cosine(matrix, query_vec)
    order = np.argsort(sims)[::-1][:top_k]
    return [
        {
            "id": docs[i]["id"],
            "text": docs[i]["text"],
            "metadata": docs[i]["metadata"],
            "score": round(float(sims[i]), 4),
        }
        for i in order
        if sims[i] > 0
    ]


def count() -> int:
    return len(_documents())


def forget(doc_id: str) -> None:
    _ensure()
    with _LOCK, sqlite3.connect(str(_db())) as conn:
        conn.execute("DELETE FROM memory_docs WHERE id = ?", (doc_id,))


# --------------------------------------------------------------------------- #
# Embedding helpers
# --------------------------------------------------------------------------- #
def _vector(texts: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not hasattr(_vector, "_vectorizer"):
        _vector._vectorizer = TfidfVectorizer(max_features=512)  # type: ignore[attr-defined]
    matrix = _vector._vectorizer.fit_transform([t or "" for t in texts]).toarray()  # type: ignore[attr-defined]
    if matrix.shape[1] < 64:
        matrix = np.hstack([matrix, np.zeros((matrix.shape[0], 64 - matrix.shape[1]))])
    return matrix


def _cosine(matrix: np.ndarray, vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vec)
    return (matrix @ vec) / np.where(norm == 0, 1.0, norm)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_docs (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    vector BLOB NOT NULL,
    metadata TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _db() -> Path:
    return get_settings().sqlite_path


def _ensure() -> None:
    with sqlite3.connect(str(_db())) as conn:
        conn.execute(_SCHEMA)


def _upsert(doc_id: str, text: str, vector: np.ndarray, metadata: dict[str, Any] | None) -> None:
    _ensure()
    with _LOCK, sqlite3.connect(str(_db())) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO memory_docs (id, text, vector, metadata) "
            "VALUES (?, ?, ?, ?)",
            (doc_id, text, _encode(vector), json.dumps(metadata or {})),
        )


def _documents() -> list[dict[str, Any]]:
    _ensure()
    with sqlite3.connect(str(_db())) as conn:
        rows = conn.execute("SELECT id, text, vector, metadata FROM memory_docs").fetchall()
    return [
        {
            "id": row[0],
            "text": row[1],
            "vector": _decode(row[2]),
            "metadata": json.loads(row[3] or "{}"),
        }
        for row in rows
    ]


def _encode(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _decode(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


__all__ = ["count", "embed_store", "forget", "search"]