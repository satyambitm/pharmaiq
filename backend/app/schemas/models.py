"""Request and response models for training jobs and model versions of uploaded datasets."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class TrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: UUID


class Job(BaseModel):
    job_id: str
    dataset_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    requested_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    attempts: int
    error: str | None = None
    version_id: str | None = None


class TrainResponse(BaseModel):
    job: Job
    created: bool
    message: str


class JobList(BaseModel):
    items: list[Job]


class VersionSummary(BaseModel):
    version_id: str
    dataset_id: str
    job_id: str
    created_at: datetime
    training_version: str
    recommended_model: str | None
    summary: dict[str, Any]


class VersionList(BaseModel):
    items: list[VersionSummary]


class VersionDetail(VersionSummary):
    results: dict[str, Any]
