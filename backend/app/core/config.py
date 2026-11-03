"""Environment-based settings. Values come from environment variables or a local `.env` file."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
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

    # Bring-your-own-data. "disabled" (default) keeps the public deployment demo-only.
    # "local" is a single-user development mode with no credentials; it is refused in production.
    # "api_key" authenticates each request with X-API-Key against API_KEYS ("owner:sha256hex,...").
    uploads_mode: Literal["disabled", "local", "api_key"] = "disabled"
    api_keys: SecretStr | None = None
    dataset_store_url: SecretStr | None = None  # defaults to DATABASE_URL
    max_upload_mb: float = Field(5.0, gt=0, le=50)
    max_upload_rows: int = Field(200_000, ge=10, le=1_000_000)
    max_datasets_per_owner: int = Field(10, ge=1, le=100)

    @model_validator(mode="after")
    def _require_url_for_postgres(self) -> "Settings":
        if self.data_backend == "postgres" and self.database_url is None:
            raise ValueError("DATABASE_URL is required when DATA_BACKEND=postgres (or set DATA_BACKEND=csv)")
        if self.uploads_mode == "local" and self.environment == "production":
            raise ValueError("UPLOADS_MODE=local has no authentication and is not allowed when ENVIRONMENT=production")
        if self.uploads_mode == "api_key":
            self.api_key_owners()  # raises on a malformed API_KEYS value
        return self

    @staticmethod
    def _driver_url(url: str) -> str:
        """Normalize `postgres://` / `postgresql://` URLs (as given by hosts) to the psycopg 3 driver."""
        for prefix in ("postgres://", "postgresql://"):
            if url.startswith(prefix):
                return "postgresql+psycopg://" + url[len(prefix):]
        return url

    @property
    def sqlalchemy_url(self) -> str:
        assert self.database_url is not None
        return self._driver_url(self.database_url.get_secret_value())

    @property
    def dataset_store_sqlalchemy_url(self) -> str | None:
        url = self.dataset_store_url or self.database_url
        return self._driver_url(url.get_secret_value()) if url else None

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    def api_key_owners(self) -> dict[str, str]:
        """Map sha256(key) hex digest -> owner id. Only hashes are configured; raw keys never reach the server config."""
        raw = self.api_keys.get_secret_value() if self.api_keys else ""
        out: dict[str, str] = {}
        for entry in filter(None, (e.strip() for e in raw.split(","))):
            owner, _, digest = entry.partition(":")
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", owner) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("API_KEYS must be comma-separated 'owner_id:sha256hex' pairs")
            out[digest] = owner
        if not out:
            raise ValueError("UPLOADS_MODE=api_key requires at least one entry in API_KEYS")
        return out


@lru_cache
def get_settings() -> Settings:
    return Settings()
