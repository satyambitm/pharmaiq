"""PharmaIQ FastAPI application entry point.

Local:   uvicorn backend.app.main:app --reload            (from the repository root)
Render:  uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from backend.app.api.routes import analytics, datasets, forecasting, health, models, optimization, segmentation
from backend.app.core.config import Settings, get_settings
from backend.app.core.errors import AppError, error_response, register_error_handlers
from backend.app.core.logging import configure_logging
from backend.app.db.connection import build_repository
from backend.app.db.datasets import DatasetStore
from ml.model_loader import BundleError, load_bundle

VERSION = "1.3.0"
logger = logging.getLogger("backend.app")
UPLOAD_PATH = "/api/v1/datasets/upload"
MULTIPART_OVERHEAD = 1024 * 1024
MAX_JSON_BODY = 1024 * 1024


def _dataset_store(settings: Settings, repository) -> tuple[DatasetStore | None, str | None]:
    """Uploads are optional: a storage problem disables them without affecting the demo endpoints."""
    if settings.uploads_mode == "disabled":
        return None, "Dataset uploads are disabled on this deployment."
    url = settings.dataset_store_sqlalchemy_url
    if url is None:
        return None, "Uploads need a database: set DATABASE_URL (or DATASET_STORE_URL)."
    try:
        engine = (repository.engine if settings.dataset_store_url is None and getattr(repository, "engine", None)
                  else create_engine(url, pool_pre_ping=True))
        store = DatasetStore(engine)
        store.ensure_schema()
        return store, None
    except AppError as exc:
        return None, exc.message
    except SQLAlchemyError as exc:  # e.g. a malformed DATASET_STORE_URL; never log the URL itself
        logger.error("dataset_store_init_failed type=%s", type(exc).__name__)
        return None, "Dataset storage could not be initialised."


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    app.state.settings = settings
    app.state.startup_problems = []
    app.state.repository = app.state.bundle = None
    try:
        app.state.repository = build_repository(settings)
        if settings.data_backend == "postgres" and settings.auto_seed_demo:
            app.state.repository.ensure_seeded(settings.sample_data_path)
    except AppError as exc:
        app.state.startup_problems.append(exc.message)
    try:
        app.state.bundle = load_bundle(settings.models_dir)
    except BundleError as exc:
        logger.error("bundle_load_failed reason=%s", exc)
        app.state.startup_problems.append("Model bundle failed to load")
    app.state.dataset_store, app.state.dataset_store_problem = _dataset_store(settings, app.state.repository)
    logger.info("startup environment=%s data_backend=%s models_loaded=%s uploads=%s store=%s", settings.environment,
                settings.data_backend, app.state.bundle is not None, settings.uploads_mode,
                app.state.dataset_store is not None)
    yield
    repo, store = app.state.repository, app.state.dataset_store
    if store is not None and store.engine is not getattr(repo, "engine", None):
        store.engine.dispose()
    if repo is not None and hasattr(repo, "engine"):
        repo.engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="PharmaIQ API",
        version=VERSION,
        description="Analytics, forecasting, segmentation and scenario-based budget optimization over "
                    "**synthetic** pharmaceutical sales (synthetic_v1). Results are illustrative only.",
        lifespan=lifespan,
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    @app.middleware("http")
    async def body_size_limit(request: Request, call_next):
        """Reject oversized bodies before they are parsed (multipart files are spooled to disk otherwise)."""
        if request.method in ("POST", "PUT", "PATCH"):
            settings: Settings = request.app.state.settings
            limit = settings.max_upload_bytes + MULTIPART_OVERHEAD if request.url.path == UPLOAD_PATH else MAX_JSON_BODY
            length = request.headers.get("content-length")
            if length is None and request.headers.get("transfer-encoding"):
                return error_response(request, 411, "length_required", "Send a Content-Length header.")
            if length is not None and (not length.isdigit() or int(length) > limit):
                return error_response(request, 413, "payload_too_large",
                                      f"The request body exceeds the {limit / 1024 / 1024:.1f} MB limit.")
        return await call_next(request)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        start = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request.state.request_id
        logger.info("request id=%s method=%s path=%s status=%s duration_ms=%.1f", request.state.request_id,
                    request.method, request.url.path, response.status_code, ms)
        return response

    register_error_handlers(app)
    for r in (health, analytics, forecasting, segmentation, optimization, datasets, models):
        app.include_router(r.router)
    return app


app = create_app()
