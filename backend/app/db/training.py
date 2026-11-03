"""Persistent training queue and model registry for uploaded datasets.

  training_jobs   queued -> running -> succeeded | failed; claimed by an out-of-process worker
  model_versions  evaluated results (JSON, served by the API) and the fitted artifact (bytes, never loaded by the API)

Claiming is a conditional UPDATE (`... WHERE status = 'queued'`), so two workers can never run the same job,
on PostgreSQL or SQLite. Every API-facing read filters on owner_id.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta

from sqlalchemy import (Column, DateTime, Engine, ForeignKey, Integer, LargeBinary, String, Table, and_, func, insert,
                        or_, select, update)

from backend.app.db.datasets import JsonType, _guard, _now, metadata

ACTIVE = ("queued", "running")

training_jobs = Table(
    "training_jobs", metadata,
    Column("job_id", String(36), primary_key=True),
    Column("owner_id", String(64), nullable=False, index=True),
    Column("dataset_id", String(36), ForeignKey("datasets.dataset_id", ondelete="CASCADE"), nullable=False, index=True),
    Column("status", String(12), nullable=False, index=True),
    Column("fingerprint", String(64), nullable=False),
    Column("requested_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=True),
    Column("heartbeat_at", DateTime(timezone=True), nullable=True),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    Column("attempts", Integer, nullable=False, default=0),
    Column("worker", String(64), nullable=True),
    Column("error", String(500), nullable=True),
    Column("version_id", String(36), nullable=True),
)

model_versions = Table(
    "model_versions", metadata,
    Column("version_id", String(36), primary_key=True),
    Column("owner_id", String(64), nullable=False, index=True),
    Column("dataset_id", String(36), ForeignKey("datasets.dataset_id", ondelete="CASCADE"), nullable=False, index=True),
    Column("job_id", String(36), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("fingerprint", String(64), nullable=False),
    Column("training_version", String(20), nullable=False),
    Column("recommended_model", String(30), nullable=True),
    Column("summary", JsonType, nullable=False),
    Column("results", JsonType, nullable=False),
    Column("artifact", LargeBinary, nullable=True),
    Column("artifact_sha256", String(64), nullable=True),
)

JOB_COLUMNS = [c for c in training_jobs.c]
VERSION_COLUMNS = [c for c in model_versions.c if c.name not in ("results", "artifact")]


def fingerprint(file_sha256: str, mapping: dict | None, training_version: str) -> str:
    """Same file + same mapping + same training code = same model; retraining would only repeat it."""
    raw = json.dumps({"file": file_sha256, "mapping": mapping or {}, "code": training_version}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


class TrainingStore:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    # ------------------------------------------------------------ API side (always owner-scoped)
    def active_job(self, owner_id: str, dataset_id: str) -> dict | None:
        with _guard("active_job"), self.engine.connect() as conn:
            row = conn.execute(select(*JOB_COLUMNS).where(training_jobs.c.owner_id == owner_id,
                                                          training_jobs.c.dataset_id == dataset_id,
                                                          training_jobs.c.status.in_(ACTIVE))).mappings().first()
        return dict(row) if row else None

    def version_for(self, owner_id: str, dataset_id: str, fp: str) -> dict | None:
        with _guard("version_for"), self.engine.connect() as conn:
            row = conn.execute(select(*VERSION_COLUMNS).where(model_versions.c.owner_id == owner_id,
                                                              model_versions.c.dataset_id == dataset_id,
                                                              model_versions.c.fingerprint == fp)).mappings().first()
        return dict(row) if row else None

    def jobs_since(self, owner_id: str, since: datetime) -> int:
        with _guard("jobs_since"), self.engine.connect() as conn:
            return int(conn.execute(select(func.count()).select_from(training_jobs).where(
                training_jobs.c.owner_id == owner_id, training_jobs.c.requested_at >= since)).scalar_one())

    def enqueue(self, owner_id: str, dataset_id: str, fp: str) -> dict:
        job = {"job_id": str(uuid.uuid4()), "owner_id": owner_id, "dataset_id": dataset_id, "status": "queued",
               "fingerprint": fp, "requested_at": _now(), "attempts": 0}
        with _guard("enqueue"), self.engine.begin() as conn:
            conn.execute(insert(training_jobs).values(**job))
        return self.get_job(owner_id, job["job_id"])  # type: ignore[return-value]

    def get_job(self, owner_id: str, job_id: str) -> dict | None:
        with _guard("get_job"), self.engine.connect() as conn:
            row = conn.execute(select(*JOB_COLUMNS).where(training_jobs.c.job_id == job_id,
                                                          training_jobs.c.owner_id == owner_id)).mappings().first()
        return dict(row) if row else None

    def list_jobs(self, owner_id: str, dataset_id: str | None, limit: int = 20) -> list[dict]:
        q = select(*JOB_COLUMNS).where(training_jobs.c.owner_id == owner_id)
        if dataset_id:
            q = q.where(training_jobs.c.dataset_id == dataset_id)
        with _guard("list_jobs"), self.engine.connect() as conn:
            rows = conn.execute(q.order_by(training_jobs.c.requested_at.desc()).limit(limit)).mappings().all()
        return [dict(r) for r in rows]

    def list_versions(self, owner_id: str, dataset_id: str | None, limit: int = 20) -> list[dict]:
        q = select(*VERSION_COLUMNS).where(model_versions.c.owner_id == owner_id)
        if dataset_id:
            q = q.where(model_versions.c.dataset_id == dataset_id)
        with _guard("list_versions"), self.engine.connect() as conn:
            rows = conn.execute(q.order_by(model_versions.c.created_at.desc()).limit(limit)).mappings().all()
        return [dict(r) for r in rows]

    def get_version(self, owner_id: str, version_id: str) -> dict | None:
        with _guard("get_version"), self.engine.connect() as conn:
            row = conn.execute(select(*VERSION_COLUMNS, model_versions.c.results).where(
                model_versions.c.version_id == version_id, model_versions.c.owner_id == owner_id)).mappings().first()
        return dict(row) if row else None

    def latest_version(self, owner_id: str, dataset_id: str) -> dict | None:
        with _guard("latest_version"), self.engine.connect() as conn:
            row = conn.execute(select(*VERSION_COLUMNS, model_versions.c.results).where(
                model_versions.c.owner_id == owner_id, model_versions.c.dataset_id == dataset_id)
                .order_by(model_versions.c.created_at.desc()).limit(1)).mappings().first()
        return dict(row) if row else None

    # ------------------------------------------------------------ worker side
    def claim(self, worker: str) -> dict | None:
        """Atomically take the oldest queued job. Returns None when the queue is empty."""
        with _guard("claim"):
            for _ in range(5):
                with self.engine.begin() as conn:
                    job_id = conn.execute(select(training_jobs.c.job_id).where(training_jobs.c.status == "queued")
                                          .order_by(training_jobs.c.requested_at).limit(1)).scalar()
                    if job_id is None:
                        return None
                    now = _now()
                    taken = conn.execute(update(training_jobs).where(training_jobs.c.job_id == job_id,
                                                                     training_jobs.c.status == "queued")
                                         .values(status="running", started_at=now, heartbeat_at=now, worker=worker[:64],
                                                 attempts=training_jobs.c.attempts + 1)).rowcount
                    if taken == 1:
                        return dict(conn.execute(select(*JOB_COLUMNS).where(training_jobs.c.job_id == job_id)).mappings().one())
        return None

    def heartbeat(self, job_id: str) -> None:
        with _guard("heartbeat"), self.engine.begin() as conn:
            conn.execute(update(training_jobs).where(training_jobs.c.job_id == job_id, training_jobs.c.status == "running")
                         .values(heartbeat_at=_now()))

    def complete(self, job: dict, results: dict, artifact: bytes, summary: dict) -> str:
        version_id = str(uuid.uuid4())
        with _guard("complete"), self.engine.begin() as conn:
            conn.execute(insert(model_versions).values(
                version_id=version_id, owner_id=job["owner_id"], dataset_id=job["dataset_id"], job_id=job["job_id"],
                created_at=_now(), fingerprint=job["fingerprint"], training_version=results["training_version"],
                recommended_model=results["forecasting"].get("recommended_model"), summary=summary, results=results,
                artifact=artifact, artifact_sha256=hashlib.sha256(artifact).hexdigest()))
            conn.execute(update(training_jobs).where(training_jobs.c.job_id == job["job_id"])
                         .values(status="succeeded", finished_at=_now(), version_id=version_id, error=None))
        return version_id

    def fail(self, job_id: str, message: str) -> None:
        with _guard("fail"), self.engine.begin() as conn:
            conn.execute(update(training_jobs).where(training_jobs.c.job_id == job_id)
                         .values(status="failed", finished_at=_now(), error=message[:500]))

    def recover_stale(self, max_silence: timedelta, max_attempts: int) -> int:
        """Requeue running jobs whose worker stopped heart-beating; give up after max_attempts."""
        cutoff = _now() - max_silence
        stale = and_(training_jobs.c.status == "running",
                     or_(training_jobs.c.heartbeat_at.is_(None), training_jobs.c.heartbeat_at < cutoff))
        with _guard("recover_stale"), self.engine.begin() as conn:
            failed = conn.execute(update(training_jobs).where(stale, training_jobs.c.attempts >= max_attempts).values(
                status="failed", finished_at=_now(), error="The worker stopped responding too many times.")).rowcount
            requeued = conn.execute(update(training_jobs).where(stale, training_jobs.c.attempts < max_attempts)
                                    .values(status="queued", worker=None)).rowcount
        return int(requeued + failed)
