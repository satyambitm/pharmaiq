"""Training jobs and model versions for uploaded datasets (owner-scoped; the demo models are unaffected)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Query

from backend.app.api.deps import get_dataset_store, get_training_store
from backend.app.core.auth import Owner, get_owner, get_settings_from_app
from backend.app.core.config import Settings
from backend.app.db.datasets import DatasetStore
from backend.app.db.training import TrainingStore
from backend.app.schemas.models import JobList, Job, TrainRequest, TrainResponse, VersionDetail, VersionList
from backend.app.services import training_service

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.post("/train", response_model=TrainResponse, status_code=202,
             summary="Queue model training for an ingested dataset (runs in a separate worker)")
def train(body: TrainRequest, owner: Owner = Depends(get_owner), datasets: DatasetStore = Depends(get_dataset_store),
          training: TrainingStore = Depends(get_training_store),
          settings: Settings = Depends(get_settings_from_app)) -> dict:
    return training_service.request_training(datasets, training, settings, owner, str(body.dataset_id))


@router.get("/jobs", response_model=JobList, summary="Your recent training jobs")
def jobs(dataset_id: UUID | None = Query(None), owner: Owner = Depends(get_owner),
         training: TrainingStore = Depends(get_training_store)) -> dict:
    return {"items": training.list_jobs(owner.owner_id, str(dataset_id) if dataset_id else None)}


@router.get("/jobs/{job_id}", response_model=Job, summary="Status of one training job")
def job(job_id: UUID, owner: Owner = Depends(get_owner), training: TrainingStore = Depends(get_training_store)) -> dict:
    return training_service.job(training, owner, str(job_id))


@router.get("/versions", response_model=VersionList, summary="Trained model versions with their evaluation summary")
def versions(dataset_id: UUID | None = Query(None), owner: Owner = Depends(get_owner),
             training: TrainingStore = Depends(get_training_store)) -> dict:
    return {"items": training.list_versions(owner.owner_id, str(dataset_id) if dataset_id else None)}


@router.get("/versions/{version_id}", response_model=VersionDetail,
            summary="Full results of a model version: metrics, forecasts, segments and explanations")
def version(version_id: UUID, owner: Owner = Depends(get_owner),
            training: TrainingStore = Depends(get_training_store)) -> dict:
    return training_service.version(training, owner, str(version_id))
