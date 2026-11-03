"""Bring-your-own-data endpoints. Owner identity comes from authentication only, never from the request body."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile

from backend.app.api.deps import get_dataset_store
from backend.app.core.auth import Owner, get_owner, get_settings_from_app
from backend.app.core.config import Settings
from backend.app.db.datasets import DatasetStore
from backend.app.schemas.datasets import (DatasetDetail, DatasetList, DeleteResponse, FieldInfo, UploadResponse,
                                          UploadSettings, ValidateRequest, ValidationResponse)
from backend.app.services import dataset_service
from src.ingest.fields import FIELDS

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])

NOTES = {
    "disabled": "Uploads are turned off on this deployment; every page uses the synthetic demo dataset.",
    "local": "Development-only local mode: no authentication, single owner. Never enable this on a public server.",
    "api_key": "Each request must send an X-API-Key issued by the operator. Datasets are private to that key's owner.",
}


@router.get("/settings", response_model=UploadSettings, summary="Whether uploads are enabled, limits and mappable fields")
def upload_settings(request: Request, settings: Settings = Depends(get_settings_from_app)) -> dict:
    return {
        "uploads_enabled": settings.uploads_mode != "disabled",
        "mode": settings.uploads_mode,
        "auth_header": "X-API-Key" if settings.uploads_mode == "api_key" else None,
        "max_upload_mb": settings.max_upload_mb,
        "max_upload_rows": settings.max_upload_rows,
        "max_datasets_per_owner": settings.max_datasets_per_owner,
        "supported_types": ["csv", "xlsx"],
        "fields": [FieldInfo(name=f.name, label=f.label, required=f.required, description=f.description)
                   for f in FIELDS.values()],
        "store_available": getattr(request.app.state, "dataset_store", None) is not None,
        "note": NOTES[settings.uploads_mode],
    }


@router.post("/upload", response_model=UploadResponse, status_code=201,
             summary="Store an uploaded CSV/XLSX and return a preview, column profile and suggested mapping")
def upload(file: UploadFile = File(...), dataset_name: str | None = Form(None, max_length=120),
           owner: Owner = Depends(get_owner), store: DatasetStore = Depends(get_dataset_store),
           settings: Settings = Depends(get_settings_from_app)) -> dict:
    content = file.file.read(settings.max_upload_bytes + 1)
    return dataset_service.upload(store, settings, owner, file.filename or "", content, dataset_name)


@router.get("/{dataset_id}/profile", response_model=UploadResponse,
            summary="Re-read the stored file: preview, column profile and suggested mapping")
def file_profile(dataset_id: UUID, owner: Owner = Depends(get_owner), store: DatasetStore = Depends(get_dataset_store),
                 settings: Settings = Depends(get_settings_from_app)) -> dict:
    return dataset_service.file_profile(store, settings, owner, str(dataset_id))


@router.post("/{dataset_id}/validate", response_model=ValidationResponse,
             summary="Apply a column mapping and return the data quality report (nothing is ingested)")
def validate(dataset_id: UUID, body: ValidateRequest, owner: Owner = Depends(get_owner),
             store: DatasetStore = Depends(get_dataset_store), settings: Settings = Depends(get_settings_from_app)) -> dict:
    return dataset_service.validate(store, settings, owner, str(dataset_id), body.model_dump())


@router.post("/{dataset_id}/activate", response_model=DatasetDetail,
             summary="Confirm the validated transformation and ingest the monthly rows")
def activate(dataset_id: UUID, owner: Owner = Depends(get_owner), store: DatasetStore = Depends(get_dataset_store),
             settings: Settings = Depends(get_settings_from_app)) -> dict:
    return dataset_service.activate(store, settings, owner, str(dataset_id))


@router.get("", response_model=DatasetList, summary="Your datasets, newest first")
def list_datasets(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
                  owner: Owner = Depends(get_owner), store: DatasetStore = Depends(get_dataset_store)) -> dict:
    return dataset_service.list_datasets(store, owner, limit, offset)


@router.get("/{dataset_id}", response_model=DatasetDetail, summary="One dataset with its latest quality report")
def detail(dataset_id: UUID, owner: Owner = Depends(get_owner), store: DatasetStore = Depends(get_dataset_store)) -> dict:
    return dataset_service.detail(store, owner, str(dataset_id))


@router.delete("/{dataset_id}", response_model=DeleteResponse, summary="Delete a dataset, its file and its rows")
def delete(dataset_id: UUID, owner: Owner = Depends(get_owner), store: DatasetStore = Depends(get_dataset_store)) -> dict:
    return dataset_service.remove(store, owner, str(dataset_id))
