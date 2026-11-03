"""Storage for uploaded datasets, kept apart from the synthetic demo table (`monthly_sales`).

Three tables, all keyed by dataset_id and scoped by owner_id:
  datasets       metadata, mapping and quality report (one row per upload)
  dataset_files  the original uploaded bytes, kept unchanged so validation can be re-run and audited
  dataset_rows   the confirmed monthly region x product panel

Every read and write takes the owner id from authentication and filters on it, so a caller can
never reach another owner's dataset by guessing its id. Files and rows live in PostgreSQL because
the Render filesystem is ephemeral. Schema changes are recorded in `schema_migrations`.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import (JSON, Column, Date, DateTime, Engine, Float, ForeignKey, Integer, LargeBinary, MetaData,
                        String, Table, delete, func, insert, select, update)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import SQLAlchemyError

from backend.app.core.errors import ServiceUnavailable
from src.ingest.fields import MONTHLY_COLUMNS

logger = logging.getLogger(__name__)
MIGRATIONS = ("001_datasets", "002_training")
JsonType = JSON().with_variant(JSONB(), "postgresql")
metadata = MetaData()

schema_migrations = Table(
    "schema_migrations", metadata,
    Column("version", String(64), primary_key=True),
    Column("applied_at", DateTime(timezone=True), nullable=False),
)

datasets = Table(
    "datasets", metadata,
    Column("dataset_id", String(36), primary_key=True),
    Column("owner_id", String(64), nullable=False, index=True),
    Column("dataset_name", String(120), nullable=False),
    Column("source_type", String(10), nullable=False),
    Column("status", String(20), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("file_name", String(255), nullable=False),
    Column("file_size_bytes", Integer, nullable=False),
    Column("file_sha256", String(64), nullable=False),
    Column("source_row_count", Integer, nullable=False),
    Column("row_count", Integer, nullable=True),
    Column("date_start", Date, nullable=True),
    Column("date_end", Date, nullable=True),
    Column("schema_mapping", JsonType, nullable=True),
    Column("schema_version", String(20), nullable=True),
    Column("quality_report", JsonType, nullable=True),
)

dataset_files = Table(
    "dataset_files", metadata,
    Column("dataset_id", String(36), ForeignKey("datasets.dataset_id", ondelete="CASCADE"), primary_key=True),
    Column("content", LargeBinary, nullable=False),
)

dataset_rows = Table(
    "dataset_rows", metadata,
    Column("dataset_id", String(36), ForeignKey("datasets.dataset_id", ondelete="CASCADE"), primary_key=True),
    Column("month_start", Date, primary_key=True),
    Column("region", String(80), primary_key=True),
    Column("product", String(80), primary_key=True),
    *[Column(c, Float, nullable=c != "units_sold") for c in MONTHLY_COLUMNS[3:-1]],
    Column("n_source_rows", Integer, nullable=False),
)

META_COLUMNS = [c for c in datasets.c if c.name not in ("quality_report",)]


def _now() -> datetime:
    return datetime.now(timezone.utc)


@contextmanager
def _guard(op: str) -> Iterator[None]:
    """Turn driver errors into a safe 503 without leaking SQL, credentials or row values."""
    try:
        yield
    except SQLAlchemyError as exc:
        logger.error("dataset_store_error op=%s type=%s", op, type(exc).__name__)
        raise ServiceUnavailable("Dataset storage is unavailable. Please retry shortly.") from exc


class DatasetStore:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def ensure_schema(self) -> None:
        """Idempotent, additive migrations: create missing tables and record each migration once."""
        from backend.app.db import training  # noqa: F401  (registers the training tables on `metadata`)

        with _guard("migration"):
            metadata.create_all(self.engine, checkfirst=True)
            with self.engine.begin() as conn:
                applied = set(conn.execute(select(schema_migrations.c.version)).scalars())
                for version in MIGRATIONS:
                    if version not in applied:
                        conn.execute(insert(schema_migrations).values(version=version, applied_at=_now()))
                        logger.info("migration_applied version=%s", version)

    def ping(self) -> None:
        with _guard("ping"), self.engine.connect() as conn:
            conn.execute(select(func.count()).select_from(schema_migrations))

    def count(self, owner_id: str) -> int:
        with _guard("count"), self.engine.connect() as conn:
            return int(conn.execute(select(func.count()).select_from(datasets)
                                    .where(datasets.c.owner_id == owner_id)).scalar_one())

    def create(self, owner_id: str, name: str, source_type: str, file_name: str, content: bytes, sha256: str,
               source_row_count: int) -> dict:
        now, dataset_id = _now(), str(uuid.uuid4())
        row = {"dataset_id": dataset_id, "owner_id": owner_id, "dataset_name": name, "source_type": source_type,
               "status": "uploaded", "created_at": now, "updated_at": now, "file_name": file_name[:255],
               "file_size_bytes": len(content), "file_sha256": sha256, "source_row_count": source_row_count}
        with _guard("create"), self.engine.begin() as conn:
            conn.execute(insert(datasets).values(**row))
            conn.execute(insert(dataset_files).values(dataset_id=dataset_id, content=content))
        return self.get(owner_id, dataset_id)  # type: ignore[return-value]

    def get(self, owner_id: str, dataset_id: str, with_report: bool = False) -> dict | None:
        cols = list(datasets.c) if with_report else META_COLUMNS
        with _guard("get"), self.engine.connect() as conn:
            row = conn.execute(select(*cols).where(datasets.c.dataset_id == dataset_id,
                                                   datasets.c.owner_id == owner_id)).mappings().first()
        return dict(row) if row else None

    def list(self, owner_id: str, limit: int, offset: int) -> tuple[list[dict], int]:
        with _guard("list"), self.engine.connect() as conn:
            total = conn.execute(select(func.count()).select_from(datasets).where(datasets.c.owner_id == owner_id)).scalar_one()
            rows = conn.execute(select(*META_COLUMNS).where(datasets.c.owner_id == owner_id)
                                .order_by(datasets.c.created_at.desc(), datasets.c.dataset_id)
                                .limit(limit).offset(offset)).mappings().all()
        return [dict(r) for r in rows], int(total)

    def file_content(self, owner_id: str, dataset_id: str) -> bytes | None:
        with _guard("file"), self.engine.connect() as conn:
            return conn.execute(select(dataset_files.c.content).join(datasets)
                                .where(datasets.c.dataset_id == dataset_id, datasets.c.owner_id == owner_id)).scalar()

    def save_validation(self, owner_id: str, dataset_id: str, mapping: dict, schema_version: str, report: dict) -> None:
        status = "validated" if report["status"] == "ready_to_ingest" else "invalid"
        with _guard("save_validation"), self.engine.begin() as conn:
            conn.execute(update(datasets).where(datasets.c.dataset_id == dataset_id, datasets.c.owner_id == owner_id)
                         .values(status=status, schema_mapping=mapping, schema_version=schema_version,
                                 quality_report=report, updated_at=_now()))

    def ingest(self, owner_id: str, dataset_id: str, monthly: pd.DataFrame) -> None:
        """Replace this dataset's monthly rows and mark it ready, in one transaction."""
        records = monthly.astype(object).where(monthly.notna(), None).to_dict("records")
        for r in records:
            r["dataset_id"] = dataset_id
            r["month_start"] = pd.Timestamp(r["month_start"]).date()
            r["n_source_rows"] = int(r["n_source_rows"])
        with _guard("ingest"), self.engine.begin() as conn:
            owned = conn.execute(select(datasets.c.dataset_id).where(datasets.c.dataset_id == dataset_id,
                                                                      datasets.c.owner_id == owner_id)).first()
            if owned is None:
                raise LookupError(dataset_id)
            conn.execute(delete(dataset_rows).where(dataset_rows.c.dataset_id == dataset_id))
            if records:
                conn.execute(insert(dataset_rows), records)
            conn.execute(update(datasets).where(datasets.c.dataset_id == dataset_id).values(
                status="ready", row_count=len(records), updated_at=_now(),
                date_start=min(r["month_start"] for r in records) if records else None,
                date_end=max(r["month_start"] for r in records) if records else None))

    def rows(self, owner_id: str, dataset_id: str) -> pd.DataFrame:
        cols = [dataset_rows.c[c] for c in MONTHLY_COLUMNS]
        with _guard("rows"), self.engine.connect() as conn:
            result = conn.execute(select(*cols).join(datasets)
                                  .where(datasets.c.dataset_id == dataset_id, datasets.c.owner_id == owner_id)
                                  .order_by(dataset_rows.c.region, dataset_rows.c.product, dataset_rows.c.month_start))
            df = pd.DataFrame(result.fetchall(), columns=MONTHLY_COLUMNS)
        df["month_start"] = pd.to_datetime(df["month_start"])
        return df

    def delete(self, owner_id: str, dataset_id: str) -> bool:
        """Remove the dataset, its file, rows, jobs and model versions. Explicit deletes so SQLite (tests) matches
        PostgreSQL cascades."""
        from backend.app.db.training import model_versions, training_jobs

        with _guard("delete"), self.engine.begin() as conn:
            owned = conn.execute(select(datasets.c.dataset_id).where(datasets.c.dataset_id == dataset_id,
                                                                      datasets.c.owner_id == owner_id)).first()
            if owned is None:
                return False
            conn.execute(delete(model_versions).where(model_versions.c.dataset_id == dataset_id))
            conn.execute(delete(training_jobs).where(training_jobs.c.dataset_id == dataset_id))
            conn.execute(delete(dataset_rows).where(dataset_rows.c.dataset_id == dataset_id))
            conn.execute(delete(dataset_files).where(dataset_files.c.dataset_id == dataset_id))
            conn.execute(delete(datasets).where(datasets.c.dataset_id == dataset_id))
        return True

