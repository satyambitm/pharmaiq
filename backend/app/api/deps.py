"""FastAPI dependencies: shared resources created once at startup and stored on `app.state`."""

from __future__ import annotations

from fastapi import Request

from backend.app.core.errors import ServiceUnavailable
from backend.app.db.connection import SalesRepository
from backend.app.db.datasets import DatasetStore
from backend.app.schemas.common import Meta
from ml.model_loader import ModelBundle


def get_repository(request: Request) -> SalesRepository:
    repo = getattr(request.app.state, "repository", None)
    if repo is None:
        raise ServiceUnavailable("Data store is not initialised")
    return repo


def get_bundle(request: Request) -> ModelBundle:
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise ServiceUnavailable("Model bundle is not loaded")
    return bundle


def get_dataset_store(request: Request) -> DatasetStore:
    store = getattr(request.app.state, "dataset_store", None)
    if store is None:
        raise ServiceUnavailable(getattr(request.app.state, "dataset_store_problem", None)
                                 or "Dataset storage is not available")
    return store


def get_meta(request: Request) -> Meta:
    repo = getattr(request.app.state, "repository", None)
    return Meta(data_backend=repo.backend_name if repo else "unavailable")
