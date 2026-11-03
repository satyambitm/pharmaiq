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

from backend.app.api.routes import analytics, forecasting, health, optimization, segmentation
from backend.app.core.config import get_settings
from backend.app.core.errors import AppError, register_error_handlers
from backend.app.core.logging import configure_logging
from backend.app.db.connection import build_repository
from ml.model_loader import BundleError, load_bundle

VERSION = "1.1.0"
logger = logging.getLogger("backend.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
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
    logger.info("startup environment=%s data_backend=%s models_loaded=%s", settings.environment,
                settings.data_backend, app.state.bundle is not None)
    yield
    repo = app.state.repository
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
    for r in (health, analytics, forecasting, segmentation, optimization):
        app.include_router(r.router)
    return app


app = create_app()
