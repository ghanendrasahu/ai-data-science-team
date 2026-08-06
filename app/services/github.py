"""Optional GitHub export of generated reports.

Uses the ``requests``-only GitHub REST API so no extra dependency is needed.
The feature is disabled (returns a clear error) when ``GITHUB_TOKEN`` /
``GITHUB_REPO`` are not configured in ``.env``, so nothing fails at startup.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

import requests

from app.config import get_settings

logger = logging.getLogger(__name__)

_API = "https://api.github.com"


def is_configured() -> bool:
    settings = get_settings()
    return bool(settings.github_token and settings.github_repo)


def publish_report(run_id: str) -> dict[str, Any]:
    """Push a run's Markdown report into ``reports/{run_id}.md`` on GitHub."""
    settings = get_settings()
    if not settings.github_token or not settings.github_repo:
        raise RuntimeError("GitHub export is not configured (set GITHUB_TOKEN and GITHUB_REPO).")

    markdown = _report_markdown(run_id)
    path = f"reports/{run_id}.md"
    headers = {
        "Authorization": f"Bearer {settings.github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    existing = requests.get(f"{_API}/repos/{settings.github_repo}/contents/{path}", headers=headers, timeout=30)
    sha = None
    if existing.status_code == 200:
        sha = existing.json().get("sha")

    body = {
        "message": f"Add report for run {run_id}",
        "content": base64.b64encode(markdown.encode("utf-8")).decode("ascii"),
        "branch": "main",
    }
    if sha:
        body["sha"] = sha

    response = requests.put(
        f"{_API}/repos/{settings.github_repo}/contents/{path}",
        json=body,
        headers=headers,
        timeout=30,
    )
    if response.status_code not in (200, 201):
        logger.warning("GitHub publish failed: %s", response.text)
        raise RuntimeError(f"GitHub publish failed: {response.status_code}")

    return {"published": True, "path": path, "html_url": f"https://github.com/{settings.github_repo}/blob/main/{path}"}


def _report_markdown(run_id: str) -> str:
    from app.agents.report import ReportAgent
    from app.services.store import get_store

    record = get_store().get_run(run_id)
    if record is None:
        raise FileNotFoundError(f"Run {run_id} not found.")
    result = record.get("result") or {}
    return result.get("report_markdown") or ReportAgent._render_report(result, "", "")


__all__ = ["is_configured", "publish_report"]