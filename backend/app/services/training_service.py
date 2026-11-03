"""Queue training for an ingested dataset and read back evaluated model versions.

The API only enqueues: no model is fitted inside a request or at startup. A deterministic fingerprint
(file hash + mapping + training code version) prevents retraining the same data, and a per-owner daily
limit bounds the queue.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from backend.app.core.auth import Owner
from backend.app.core.config import Settings
from backend.app.core.errors import AppError, Conflict, NotFound
from backend.app.db.datasets import DatasetStore, _now
from backend.app.db.training import TrainingStore, fingerprint
from src.training import TRAINING_VERSION

logger = logging.getLogger(__name__)


def request_training(datasets: DatasetStore, training: TrainingStore, settings: Settings, owner: Owner,
                     dataset_id: str) -> dict:
    meta = datasets.get(owner.owner_id, dataset_id)
    if meta is None:
        raise NotFound("Dataset not found")
    if meta["status"] != "ready":
        raise Conflict("Confirm ingestion of this dataset before training models on it.")
    active = training.active_job(owner.owner_id, dataset_id)
    if active is not None:
        return {"job": active, "created": False, "message": "A training job for this dataset is already queued or running."}
    fp = fingerprint(meta["file_sha256"], meta["schema_mapping"], TRAINING_VERSION)
    existing = training.version_for(owner.owner_id, dataset_id, fp)
    if existing is not None:
        raise Conflict(f"Models were already trained on exactly this data (version {existing['version_id']}). "
                       "Training again would reproduce the same result.")
    if training.jobs_since(owner.owner_id, _now() - timedelta(days=1)) >= settings.max_training_jobs_per_day:
        raise AppError(429, "rate_limited", f"At most {settings.max_training_jobs_per_day} training requests per day.")
    job = training.enqueue(owner.owner_id, dataset_id, fp)
    logger.info("training_queued owner=%s dataset=%s job=%s", owner.owner_id, dataset_id, job["job_id"])
    return {"job": job, "created": True,
            "message": "Queued. Training runs in a separate worker process; check the job status for progress."}


def job(training: TrainingStore, owner: Owner, job_id: str) -> dict:
    found = training.get_job(owner.owner_id, job_id)
    if found is None:
        raise NotFound("Job not found")
    return found


def version(training: TrainingStore, owner: Owner, version_id: str) -> dict:
    found = training.get_version(owner.owner_id, version_id)
    if found is None:
        raise NotFound("Model version not found")
    return found
