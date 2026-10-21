"""Centralised application configuration.

Values are loaded from environment variables (`.env` supported) and validated
with pydantic-settings. Nothing here reads secrets out of the codebase.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime settings for the application.

    No secret values are hardcoded here; they all come from the environment.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AI Data Science Team"
    app_version: str = "0.1.0"

    llm_provider: str = ""

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-latest"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    upload_dir: str = "uploads"
    model_dir: str = "models"
    report_dir: str = "reports"
    deploy_dir: str = "deployments"
    sqlite_db: str = "ai_ds_team.db"

    feature_selection: bool = True
    max_features: int = 0

    # Explicit CORS allowlist: "*" is rejected by browsers when credentials are
    # in play; never re-add it here.
    backend_origins: str = (
        "http://localhost:8000,http://localhost:5173,"
        "http://localhost:3000,http://localhost:8501"
    )
    # When set, every mutating endpoint requires the `X-API-Key` header.
    # Empty string = auth disabled (development convenience).
    api_key: str = ""

    github_token: str = ""
    github_repo: str = ""
    mlflow_tracking_uri: str = ""

    @property
    def upload_dir_path(self) -> Path:
        return self._ensure(PROJECT_ROOT / self.upload_dir)

    @property
    def model_dir_path(self) -> Path:
        return self._ensure(PROJECT_ROOT / self.model_dir)

    @property
    def report_dir_path(self) -> Path:
        return self._ensure(PROJECT_ROOT / self.report_dir)

    @property
    def deploy_dir_path(self) -> Path:
        return self._ensure(PROJECT_ROOT / self.deploy_dir)

    @property
    def backend_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.backend_origins.split(",") if origin.strip()]

    @property
    def sqlite_path(self) -> Path:
        return PROJECT_ROOT / self.sqlite_db

    @staticmethod
    def _ensure(path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()