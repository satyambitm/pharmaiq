"""SQLAlchemy engine helpers for the local PostgreSQL database."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import OperationalError

from src import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def get_engine(url: str | None = None) -> Engine:
    """Return a cached engine for `url` (defaults to DATABASE_URL)."""
    return create_engine(url or config.DATABASE_URL, pool_pre_ping=True, future=True)


def db_available(url: str | None = None) -> bool:
    """True if a connection can be opened; used for graceful skips and dashboard fallbacks."""
    try:
        with get_engine(url).connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except OperationalError as exc:
        logger.warning("Database unavailable: %s", str(exc).splitlines()[0])
        return False


def run_sql_file(path: Path, engine: Engine | None = None) -> None:
    """Execute a multi-statement SQL file (no parameters) in one transaction."""
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.exec_driver_sql(path.read_text(encoding="utf-8"))
    logger.info("Executed %s", path.name)
