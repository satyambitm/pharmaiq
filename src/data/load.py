"""Idempotent PostgreSQL ingestion of the validated clean panel, with reconciliation.

Strategy: inside one transaction, upsert dimension rows, delete this dataset version's
fact rows, and re-insert them. Re-running produces identical counts and totals.
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, text

from src import config
from src.data.database import get_engine, run_sql_file
from src.data.generate import CSV_COLUMNS, MONEY_COLUMNS
from src.data.validate import DataValidationError, read_raw, run_checks

logger = logging.getLogger(__name__)

INSERT_SQL = text(f"""
INSERT INTO monthly_sales ({", ".join(CSV_COLUMNS)})
VALUES ({", ".join(":" + c for c in CSV_COLUMNS)})
""")


def _records(df: pd.DataFrame) -> list[dict]:
    """Convert rows to DB-ready dicts; money stays exact via Decimal."""
    out = []
    for row in df.to_dict("records"):
        for c in MONEY_COLUMNS:
            row[c] = Decimal(row[c])
        row["month_start"] = pd.Timestamp(row["month_start"]).date()
        row["units_sold"] = int(row["units_sold"])
        row["active_accounts"] = int(row["active_accounts"])
        row["competitor_index"] = float(row["competitor_index"])
        row["regional_demand_index"] = float(row["regional_demand_index"])
        row["campaign_flag"] = bool(row["campaign_flag"])
        out.append(row)
    return out


def load_clean(engine: Engine | None = None, csv_path: Path | None = None,
               report_path: Path | None = config.METRICS_DIR / "load_reconciliation.json") -> dict:
    """Create schema/views and load the clean CSV; return a reconciliation report."""
    engine = engine or get_engine()
    df = read_raw(csv_path or config.CLEAN_CSV)
    checks = run_checks(df)
    if not checks["passed"]:
        raise DataValidationError("Refusing to load: clean CSV failed validation")

    run_sql_file(config.SQL_DIR / "schema.sql", engine)
    with engine.begin() as conn:
        conn.execute(
            text("""INSERT INTO regions (region_id, region_name) VALUES (:id, :name)
                    ON CONFLICT (region_id) DO UPDATE SET region_name = EXCLUDED.region_name"""),
            [{"id": k, "name": v} for k, v in config.REGIONS.items()],
        )
        conn.execute(
            text("""INSERT INTO products (product_id, product_name, therapeutic_group) VALUES (:id, :name, :grp)
                    ON CONFLICT (product_id) DO UPDATE
                    SET product_name = EXCLUDED.product_name, therapeutic_group = EXCLUDED.therapeutic_group"""),
            [{"id": k, "name": v, "grp": config.THERAPEUTIC_GROUP[k]} for k, v in config.PRODUCTS.items()],
        )
        deleted = conn.execute(text("DELETE FROM monthly_sales WHERE data_source = :ds"),
                               {"ds": config.DATA_SOURCE}).rowcount
        conn.execute(INSERT_SQL, _records(df))
    run_sql_file(config.SQL_DIR / "views.sql", engine)

    report = reconcile(df, engine)
    report["rows_deleted_before_insert"] = int(deleted)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Loaded rows=%d (deleted %d first); reconciled=%s", report["db_rows"], deleted, report["reconciled"])
    if not report["reconciled"]:
        raise DataValidationError(f"Warehouse totals do not match CSV: {report}")
    return report


def reconcile(df: pd.DataFrame, engine: Engine) -> dict:
    """Compare row count and exact money totals between the CSV and the warehouse."""
    csv_totals = {c: sum(Decimal(v) for v in df[c]) for c in ["gross_revenue_inr", "discount_inr", "net_revenue_inr", "marketing_spend_inr"]}
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT COUNT(*) AS n, SUM(units_sold) AS units, SUM(gross_revenue_inr) AS gross_revenue_inr,
                   SUM(discount_inr) AS discount_inr, SUM(net_revenue_inr) AS net_revenue_inr,
                   SUM(marketing_spend_inr) AS marketing_spend_inr
            FROM monthly_sales WHERE data_source = :ds"""), {"ds": config.DATA_SOURCE}).mappings().one()
    report = {
        "csv_rows": int(len(df)),
        "db_rows": int(row["n"]),
        "csv_units": int(df["units_sold"].sum()),
        "db_units": int(row["units"]),
        "totals": {c: {"csv": str(csv_totals[c]), "db": str(row[c])} for c in csv_totals},
    }
    report["reconciled"] = bool(
        report["csv_rows"] == report["db_rows"]
        and report["csv_units"] == report["db_units"]
        and all(csv_totals[c] == row[c] for c in csv_totals)
    )
    return report


def main() -> None:
    config.setup_logging()
    config.ensure_dirs()
    load_clean()


if __name__ == "__main__":
    main()
