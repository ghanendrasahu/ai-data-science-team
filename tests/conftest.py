"""Test session bootstrap.

Redirects every writable app location (SQLite, uploads, models, reports,
deployments) into a temp dir so the suite never touches repo-root user data.
Env vars must be set BEFORE importing ``app.main`` (settings are read at import).
"""

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="ai_ds_team_tests_"))

os.environ["SQLITE_DB"] = str(_TMP / "ai_ds_team.db")
os.environ["UPLOAD_DIR"] = str(_TMP / "uploads")
os.environ["MODEL_DIR"] = str(_TMP / "models")
os.environ["REPORT_DIR"] = str(_TMP / "reports")
os.environ["DEPLOY_DIR"] = str(_TMP / "deployments")

# Never let a developer's real .env turn tests into live LLM calls.
os.environ.setdefault("LLM_PROVIDER", "")
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ["API_KEY"] = ""


