"""Data access: a small pooled PostgreSQL repository, with a bundled-CSV fallback.

Both repositories return the same typed DataFrame, so services contain one copy of the
business logic regardless of where the rows come from.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Protocol

import pandas as pd
from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import SQLAlchemyError

from backend.app.core.config import Settings
from backend.app.core.errors import ServiceUnavailable
from backend.app.db import queries
from src.config import DATA_SOURCE
from src.data.load import load_clean
from src.data.validate import read_raw, to_panel

logger = logging.getLogger(__name__)
MONEY = ["unit_price_inr", "gross_revenue_inr", "discount_inr", "net_revenue_inr", "marketing_spend_inr"]


class SalesRepository(Protocol):
    backend_name: str

    def fetch_sales(self, start: date | None = None, end: date | None = None,
                    region_ids: list[str] | None = None, product_ids: list[str] | None = None) -> pd.DataFrame: ...

    def window(self) -> dict: ...

    def ping(self) -> None: ...


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    df["month_start"] = pd.to_datetime(df["month_start"])
    for c in MONEY:
        df[c] = df[c].astype(float)
    df["campaign_flag"] = df["campaign_flag"].astype(bool)
    return df


class PostgresRepository:
    backend_name = "postgres"

    def __init__(self, settings: Settings) -> None:
        self.engine: Engine = create_engine(
            settings.sqlalchemy_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout_seconds,
            pool_pre_ping=True,
            pool_recycle=1800,
            connect_args={"connect_timeout": settings.db_connect_timeout_seconds},
        )

    def _read(self, stmt, params: dict) -> pd.DataFrame:
        try:
            with self.engine.connect() as conn:
                return pd.read_sql(stmt, conn, params=params)
        except SQLAlchemyError as exc:
            logger.error("database_error type=%s", type(exc).__name__)
            raise ServiceUnavailable("Database is unavailable. Please retry shortly.") from exc

    def fetch_sales(self, start=None, end=None, region_ids=None, product_ids=None) -> pd.DataFrame:
        df = self._read(queries.SALES_ROWS, {"data_source": DATA_SOURCE, "start_date": start, "end_date": end,
                                             "region_ids": region_ids, "product_ids": product_ids})
        return _typed(df)

    def window(self) -> dict:
        row = self._read(queries.DATA_WINDOW, {"data_source": DATA_SOURCE}).iloc[0]
        return {"first_month": pd.Timestamp(row["first_month"]), "last_month": pd.Timestamp(row["last_month"]),
                "n_rows": int(row["n_rows"])}

    def ping(self) -> None:
        self._read(queries.PING, {})

    def ensure_seeded(self, csv_path: Path) -> bool:
        """Load the bundled demo CSV if the sales table is missing or empty. Returns True if it loaded."""
        try:
            with self.engine.connect() as conn:
                exists = conn.execute(queries.TABLE_EXISTS).scalar()
                count = conn.execute(queries.COUNT_ROWS, {"data_source": DATA_SOURCE}).scalar() if exists else 0
            if count:
                return False
            report = load_clean(self.engine, csv_path=csv_path, report_path=None)
        except SQLAlchemyError as exc:
            logger.error("demo_seed_failed type=%s", type(exc).__name__)
            raise ServiceUnavailable("Database is unavailable. Please retry shortly.") from exc
        logger.info("demo_seed_loaded rows=%s reconciled=%s", report["db_rows"], report["reconciled"])
        return True


class CsvRepository:
    """Fallback for free tiers without a database: serves the bundled clean CSV (same rows as Postgres)."""

    backend_name = "csv"

    def __init__(self, settings: Settings) -> None:
        if not settings.sample_data_path.exists():
            raise ServiceUnavailable("Bundled sample data not found")
        self.df = to_panel(read_raw(settings.sample_data_path)).drop(columns=["data_source"])

    def fetch_sales(self, start=None, end=None, region_ids=None, product_ids=None) -> pd.DataFrame:
        d = self.df
        mask = pd.Series(True, index=d.index)
        if start is not None:
            mask &= d["month_start"] >= pd.Timestamp(start)
        if end is not None:
            mask &= d["month_start"] <= pd.Timestamp(end)
        if region_ids:
            mask &= d["region_id"].isin(region_ids)
        if product_ids:
            mask &= d["product_id"].isin(product_ids)
        return d[mask].sort_values(["region_id", "product_id", "month_start"]).reset_index(drop=True).copy()

    def window(self) -> dict:
        return {"first_month": self.df["month_start"].min(), "last_month": self.df["month_start"].max(), "n_rows": len(self.df)}

    def ping(self) -> None:
        return None


def build_repository(settings: Settings) -> SalesRepository:
    return PostgresRepository(settings) if settings.data_backend == "postgres" else CsvRepository(settings)
