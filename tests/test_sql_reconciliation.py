"""Integration tests against the local PostgreSQL (skipped if it is not reachable)."""

import pytest
from sqlalchemy import text

from src import config
from src.data.database import db_available, get_engine

pytestmark = pytest.mark.skipif(
    not (config.CLEAN_CSV.exists() and db_available()),
    reason="needs generated data/raw CSV and a running PostgreSQL (docker compose up -d)",
)


def test_loader_idempotency():
    from src.data.load import load_clean

    first = load_clean()
    second = load_clean()
    assert first["db_rows"] == second["db_rows"] == 1080
    assert first["totals"] == second["totals"]
    assert second["reconciled"]


def test_sql_matches_pandas_aggregates():
    import pandas as pd

    from src.analytics.sql_reporting import reconcile_with_pandas
    from src.data.validate import read_raw, to_panel

    checks = reconcile_with_pandas(to_panel(read_raw(config.CLEAN_CSV)))
    assert all(checks.values()), checks


def test_database_rejects_unreconciled_row():
    from sqlalchemy.exc import IntegrityError

    with get_engine().connect() as conn:
        trans = conn.begin()
        try:
            with pytest.raises(IntegrityError):
                conn.execute(text("""
                    INSERT INTO monthly_sales VALUES ('2030-01-01','R01','P01',10,100.00,1000.00,10.00,999.00,
                                                      0,1,1.0,1.0,false,'synthetic_v1')"""))
        finally:
            trans.rollback()
