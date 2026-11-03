from __future__ import annotations

from pydantic import BaseModel


class Liveness(BaseModel):
    status: str
    version: str


class Readiness(BaseModel):
    status: str
    data_backend: str
    data_rows: int | None = None
    data_window: str | None = None
    models_loaded: bool
    model_versions: dict[str, str] = {}
    problems: list[str] = []
