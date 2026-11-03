"""Who is calling. Every dataset query is scoped to the returned owner id, never to a client-supplied id."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

from fastapi import Header, Request

from backend.app.core.config import Settings
from backend.app.core.errors import AppError, Unauthorized

LOCAL_OWNER = "local-dev"


@dataclass(frozen=True)
class Owner:
    owner_id: str
    auth_method: str


def get_settings_from_app(request: Request) -> Settings:
    return request.app.state.settings


def get_owner(request: Request, x_api_key: str | None = Header(None, alias="X-API-Key")) -> Owner:
    settings = get_settings_from_app(request)
    if settings.uploads_mode == "disabled":
        raise AppError(403, "uploads_disabled", "Dataset uploads are disabled on this deployment.")
    if settings.uploads_mode == "local":
        return Owner(LOCAL_OWNER, "local-development")
    if not x_api_key:
        raise Unauthorized()
    digest = hashlib.sha256(x_api_key.encode("utf-8")).hexdigest()
    owner = None
    for known, owner_id in settings.api_key_owners().items():
        if hmac.compare_digest(known, digest):
            owner = owner_id
    if owner is None:
        raise Unauthorized("The API key is not valid")
    return Owner(owner, "api_key")
