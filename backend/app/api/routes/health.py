from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from backend.app.core.errors import AppError
from backend.app.schemas.health import Liveness, Readiness

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/health", response_model=Liveness, summary="Liveness: the process is up (no dependencies checked)")
def health(request: Request) -> Liveness:
    return Liveness(status="ok", version=request.app.version)


@router.get("/health/ready", response_model=Readiness, summary="Readiness: data store reachable and models loaded",
            responses={503: {"model": Readiness}})
def ready(request: Request):
    state = request.app.state
    problems = list(getattr(state, "startup_problems", []))
    repo, bundle = getattr(state, "repository", None), getattr(state, "bundle", None)
    body = Readiness(status="ready", data_backend=repo.backend_name if repo else "unavailable",
                     models_loaded=bundle is not None)
    if repo is not None:
        try:
            w = repo.window()
            body.data_rows = w["n_rows"]
            body.data_window = f"{w['first_month']:%Y-%m}..{w['last_month']:%Y-%m}"
        except AppError as exc:
            problems.append(exc.message)
    if bundle is not None:
        body.model_versions = {"xgboost": bundle.xgb_meta["model_version"], "prophet": bundle.prophet_meta["model_version"]}
    body.problems = problems
    if problems or repo is None or bundle is None:
        body.status = "not_ready"
        return JSONResponse(status_code=503, content=body.model_dump())
    return body
