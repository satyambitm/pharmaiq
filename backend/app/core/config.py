"""Environment-based settings. Values come from environment variables or a local `.env` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """All runtime configuration for the API. Never log `database_url`."""

    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "PharmaIQ API"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    data_backend: Literal["postgres", "csv"] = "postgres"
    database_url: SecretStr | None = None
    db_pool_size: int = 3
    db_max_overflow: int = 2
    db_pool_timeout_seconds: int = 10
    db_connect_timeout_seconds: int = 10
    auto_seed_demo: bool = False

    models_dir: Path = ROOT / "models"
    sample_data_path: Path = ROOT / "data" / "sample" / "synthetic_sales_clean.csv"

    @model_validator(mode="after")
    def _require_url_for_postgres(self) -> "Settings":
        if self.data_backend == "postgres" and self.database_url is None:
            raise ValueError("DATABASE_URL is required when DATA_BACKEND=postgres (or set DATA_BACKEND=csv)")
        return self

    @property
    def sqlalchemy_url(self) -> str:
        """Normalize `postgres://` / `postgresql://` URLs (as given by hosts) to the psycopg 3 driver."""
        assert self.database_url is not None
        url = self.database_url.get_secret_value()
        for prefix in ("postgres://", "postgresql://"):
            if url.startswith(prefix):
                return "postgresql+psycopg://" + url[len(prefix):]
        return url


@lru_cache
def get_settings() -> Settings:
    return Settings()
