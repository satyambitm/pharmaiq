"""Run named queries from sql/analysis_queries.sql and reconcile them with pandas."""

from __future__ import annotations

import logging
import re

import pandas as pd
from sqlalchemy import Engine, text

from src import config
from src.data.database import get_engine

logger = logging.getLogger(__name__)
NAME_RE = re.compile(r"^-- name:\s*(\S+)\s*$", re.MULTILINE)


def load_queries() -> dict[str, dict[str, str]]:
    """Parse `-- name:` blocks into {name: {"question": ..., "sql": ...}}."""
    raw = (config.SQL_DIR / "analysis_queries.sql").read_text(encoding="utf-8")
    parts = NAME_RE.split(raw)
    queries = {}
    for name, body in zip(parts[1::2], parts[2::2]):
        q = re.search(r"^-- question:\s*(.+)$", body, re.MULTILINE)
        queries[name] = {"question": q.group(1).strip() if q else "", "sql": body.strip()}
    return queries


def run_query(name: str, engine: Engine | None = None) -> pd.DataFrame:
    """Execute one named query and return its result as a DataFrame."""
    sql = load_queries()[name]["sql"]
    with (engine or get_engine()).connect() as conn:
        return pd.read_sql(text(sql), conn)


def export_all(engine: Engine | None = None) -> dict[str, int]:
    """Run every query, save CSVs to artifacts/reports, return row counts."""
    config.ensure_dirs()
    counts = {}
    for name in load_queries():
        df = run_query(name, engine)
        df.to_csv(config.REPORTS_DIR / f"sql_{name}.csv", index=False)
        counts[name] = len(df)
        logger.info("%s -> %d rows", name, len(df))
    return counts


def reconcile_with_pandas(panel: pd.DataFrame, engine: Engine | None = None) -> dict[str, bool]:
    """Cross-check a few SQL outputs against independent pandas aggregations."""
    engine = engine or get_engine()
    checks = {}
    q1 = run_query("q01_monthly_totals", engine)
    pd_monthly = panel.groupby("month_start").agg(units_sold=("units_sold", "sum"), net=("net_revenue_inr", "sum")).reset_index()
    checks["q01_units_match"] = bool((q1["units_sold"].astype(int).to_numpy() == pd_monthly["units_sold"].to_numpy()).all())
    checks["q01_net_match"] = bool(((q1["net_revenue_inr"].astype(float) - pd_monthly["net"]).abs() < 0.01).all())
    q3 = run_query("q03_region_revenue_share", engine)
    checks["q03_shares_sum_100"] = bool(abs(q3["revenue_share_pct"].astype(float).sum() - 100) < 0.05)
    q10 = run_query("q10_missing_panel_keys", engine)
    checks["q10_no_missing_keys"] = bool(len(q10) == 0)
    return checks
