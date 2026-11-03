"""Application errors and handlers that return a consistent JSON shape without internal details."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    """An expected failure with a safe, user-facing message."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code, self.code, self.message = status_code, code, message


class ServiceUnavailable(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(503, "service_unavailable", message)


class InvalidRequest(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(422, "invalid_request", message)


def _body(request: Request, code: str, message: str, details: object | None = None) -> dict:
    body = {"error": {"code": code, "message": message, "request_id": getattr(request.state, "request_id", None)}}
    if details is not None:
        body["error"]["details"] = details
    return body


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.warning("app_error code=%s path=%s message=%s", exc.code, request.url.path, exc.message)
        return JSONResponse(status_code=exc.status_code, content=_body(request, exc.code, exc.message))

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [{"loc": list(e.get("loc", [])), "msg": e.get("msg")} for e in exc.errors()]
        return JSONResponse(status_code=422, content=_body(request, "validation_error", "Invalid request parameters", details))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error path=%s", request.url.path)
        return JSONResponse(status_code=500, content=_body(request, "internal_error", "Unexpected server error"))
