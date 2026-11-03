"""Data-quality checks, dirty-copy repair rules and the audit report.

The clean CSV must pass every check before it becomes the processed modeling panel.
The dirty CSV is only used to demonstrate detection and repair; it is never used for modeling.
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.data.generate import CSV_COLUMNS, MONEY_COLUMNS

logger = logging.getLogger(__name__)
KEY = ["month_start", "region_id", "product_id"]


class DataValidationError(ValueError):
    """Raised when a dataset that must be clean fails a quality check."""


def read_raw(path: Path) -> pd.DataFrame:
    """Read a raw CSV keeping money columns as strings so they can be converted exactly."""
    return pd.read_csv(path, dtype={c: str for c in MONEY_COLUMNS} | {"month_start": str})


def to_paise(values: pd.Series) -> pd.Series:
    """Convert 2-decimal INR strings to integer paise (nullable) using Decimal arithmetic."""
    def conv(v: object) -> object:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return pd.NA
        try:
            return int((Decimal(str(v)) * 100).to_integral_value())
        except InvalidOperation:
            return pd.NA
    return values.map(conv).astype("Int64")


def expected_keys() -> pd.DataFrame:
    """Full calendar x region x product panel that must exist."""
    months = pd.date_range(config.START_MONTH, periods=config.N_MONTHS, freq="MS").strftime("%Y-%m-%d")
    idx = pd.MultiIndex.from_product([months, list(config.REGIONS), list(config.PRODUCTS)], names=KEY)
    return idx.to_frame(index=False)


def run_checks(df: pd.DataFrame) -> dict:
    """Run all quality checks; return a JSON-serializable report with per-check counts."""
    report: dict = {"n_rows": int(len(df)), "expected_rows": config.N_MONTHS * len(config.REGIONS) * len(config.PRODUCTS)}
    missing_cols = [c for c in CSV_COLUMNS if c not in df.columns]
    report["missing_columns"] = missing_cols
    if missing_cols:
        report["passed"] = False
        return report

    dates = pd.to_datetime(df["month_start"], errors="coerce")
    report["unparseable_dates"] = int(dates.isna().sum())
    report["non_month_start_dates"] = int((dates.dt.day != 1).sum())
    report["duplicate_keys"] = int(df.duplicated(KEY).sum())
    report["missing_values"] = {c: int(v) for c, v in df[CSV_COLUMNS].isna().sum().items() if v > 0}

    paise = {c: to_paise(df[c]) for c in MONEY_COLUMNS}
    report["negative_values"] = {
        c: int((s < 0).sum()) for c, s in
        {**{k: paise[k] for k in MONEY_COLUMNS if k != "unit_price_inr"},
         "units_sold": df["units_sold"], "active_accounts": df["active_accounts"]}.items()
        if int((s < 0).sum()) > 0
    }
    report["nonpositive_price"] = int((paise["unit_price_inr"] <= 0).sum())
    report["discount_exceeds_gross"] = int((paise["discount_inr"] > paise["gross_revenue_inr"]).sum())
    gross_calc = df["units_sold"].astype("Int64") * paise["unit_price_inr"]
    report["gross_mismatch"] = int((gross_calc != paise["gross_revenue_inr"]).fillna(True).sum())
    net_calc = paise["gross_revenue_inr"] - paise["discount_inr"]
    report["net_mismatch"] = int((net_calc != paise["net_revenue_inr"]).fillna(True).sum())

    norm = df.assign(month_start=dates.dt.to_period("M").dt.to_timestamp().dt.strftime("%Y-%m-%d"))
    exp = expected_keys()
    merged = exp.merge(norm[KEY].drop_duplicates(), on=KEY, how="left", indicator=True)
    missing_keys = merged[merged["_merge"] == "left_only"][KEY]
    report["missing_panel_keys"] = int(len(missing_keys))
    report["missing_panel_key_examples"] = missing_keys.head(10).to_dict("records")
    report["unknown_region_ids"] = sorted(set(df["region_id"]) - set(config.REGIONS))
    report["unknown_product_ids"] = sorted(set(df["product_id"]) - set(config.PRODUCTS))
    report["unexpected_data_source"] = sorted(set(df["data_source"]) - {config.DATA_SOURCE})

    failures = [
        report["unparseable_dates"], report["non_month_start_dates"], report["duplicate_keys"],
        sum(report["missing_values"].values()), sum(report["negative_values"].values()),
        report["nonpositive_price"], report["discount_exceeds_gross"], report["gross_mismatch"],
        report["net_mismatch"], report["missing_panel_keys"], len(report["unknown_region_ids"]),
        len(report["unknown_product_ids"]), len(report["unexpected_data_source"]),
        int(report["n_rows"] != report["expected_rows"]),
    ]
    report["passed"] = bool(sum(failures) == 0)
    return report


def repair(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply documented, history-only repair rules to the dirty copy.

    Rules: normalize dates to month start; drop duplicate keys (keep first); replace
    negative spend with the same series' previous-month spend; forward-fill missing
    competitor_index within series; recompute net = gross - discount. Missing panel rows are
    reported, never fabricated.
    """
    actions: list[str] = []
    out = df.copy()
    dates = pd.to_datetime(out["month_start"])
    n_dates = int((dates.dt.day != 1).sum())
    out["month_start"] = dates.dt.to_period("M").dt.to_timestamp().dt.strftime("%Y-%m-%d")
    actions.append(f"normalized {n_dates} non-month-start dates to month start")

    n_before = len(out)
    out = out.drop_duplicates(KEY, keep="first")
    actions.append(f"dropped {n_before - len(out)} duplicate key rows (kept first occurrence)")

    out = out.sort_values(["region_id", "product_id", "month_start"]).reset_index(drop=True)
    spend = to_paise(out["marketing_spend_inr"])
    neg = spend < 0
    spend = spend.mask(neg)
    spend = spend.groupby([out["region_id"], out["product_id"]]).ffill()
    out["marketing_spend_inr"] = [f"{int(p) // 100}.{int(p) % 100:02d}" if pd.notna(p) else np.nan for p in spend]
    actions.append(f"replaced {int(neg.sum())} negative marketing_spend values with previous-month spend")

    n_missing = int(out["competitor_index"].isna().sum())
    out["competitor_index"] = out.groupby(["region_id", "product_id"])["competitor_index"].ffill()
    actions.append(f"forward-filled {n_missing} missing competitor_index values within series "
                   f"({int(out['competitor_index'].isna().sum())} remain)")

    gross = to_paise(out["gross_revenue_inr"])
    disc = to_paise(out["discount_inr"])
    net = to_paise(out["net_revenue_inr"])
    bad = (gross - disc) != net
    out["net_revenue_inr"] = [f"{int(p) // 100}.{int(p) % 100:02d}" for p in (gross - disc)]
    actions.append(f"recomputed net_revenue_inr for {int(bad.sum())} reconciliation mismatches")

    out = out.sort_values(KEY).reset_index(drop=True)
    return out, actions


def to_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Typed analysis panel: datetime month, float INR, categorical ids."""
    panel = df.copy()
    panel["month_start"] = pd.to_datetime(panel["month_start"])
    for c in MONEY_COLUMNS:
        panel[c] = to_paise(panel[c]).astype("float64") / 100.0
    panel["campaign_flag"] = panel["campaign_flag"].astype(bool)
    return panel.sort_values(["region_id", "product_id", "month_start"]).reset_index(drop=True)


def validate_and_build() -> dict:
    """Validate clean data (must pass), audit + repair the dirty copy, write outputs."""
    config.ensure_dirs()
    clean = read_raw(config.CLEAN_CSV)
    clean_report = run_checks(clean)
    if not clean_report["passed"]:
        raise DataValidationError(f"Clean dataset failed validation: {clean_report}")
    to_panel(clean).to_parquet(config.PANEL_PARQUET, index=False)

    dirty = read_raw(config.DIRTY_CSV)
    dirty_report = run_checks(dirty)
    repaired, actions = repair(dirty)
    repaired.to_csv(config.REPAIRED_DIRTY_CSV, index=False, lineterminator="\n")
    repaired_report = run_checks(repaired)

    both = repaired.merge(clean, on=KEY, suffixes=("_rep", "_clean"))
    diff_cols = {}
    for c in [c for c in CSV_COLUMNS if c not in KEY]:
        a, b = both[f"{c}_rep"].astype(str), both[f"{c}_clean"].astype(str)
        n = int((a != b).sum())
        if n:
            diff_cols[c] = n

    report = {
        "clean": clean_report,
        "dirty_before_repair": dirty_report,
        "repair_actions": actions,
        "dirty_after_repair": repaired_report,
        "repaired_vs_clean_cell_differences": diff_cols,
        "note": "Modeling uses the clean panel only. Repaired dirty data is for audit practice.",
    }
    (config.METRICS_DIR / "data_quality_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Clean passed=%s rows=%d | dirty dupes=%d missing_keys=%d net_mismatch=%d | after repair passed=%s",
                clean_report["passed"], clean_report["n_rows"], dirty_report["duplicate_keys"],
                dirty_report["missing_panel_keys"], dirty_report["net_mismatch"], repaired_report["passed"])
    return report


def main() -> None:
    config.setup_logging()
    validate_and_build()


if __name__ == "__main__":
    main()
