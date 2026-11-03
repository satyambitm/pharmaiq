"""Leakage-safe one-month-ahead feature matrix.

Each row is labeled by its *target month* T for a (region, product) series. The forecast
origin is O = T - 1 month. Every feature uses only data observed by the end of O:

    lag_k          = units_sold at month T - k          (lag_1 = last observed month O)
    roll_mean_w    = mean of units over months T-w .. T-1  (shift(1) then rolling(w))
    roll_std_w     = sample std over the same window
    *_origin       = origin-month value of spend, price, competitor index, accounts, campaign
    spend_sum_3    = marketing spend over months T-3 .. T-1
    month_sin/cos  = target-month calendar position (known in advance)

The target-month realized spend, price, revenue and competitor index are never used.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

GROUP = ["region_id", "product_id"]
LAGS = (1, 2, 3, 6, 12)
WINDOWS = (3, 6, 12)
CATEGORICAL_FEATURES = ["region_id", "product_id"]
NUMERIC_FEATURES = (
    [f"lag_{k}" for k in LAGS]
    + [f"roll_mean_{w}" for w in WINDOWS]
    + [f"roll_std_{w}" for w in WINDOWS]
    + ["spend_origin", "spend_sum_3", "price_origin", "competitor_origin", "accounts_origin",
       "campaign_origin", "month_sin", "month_cos"]
)
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "y"


def build_features(panel: pd.DataFrame, drop_incomplete: bool = True) -> pd.DataFrame:
    """Build the supervised matrix from the typed monthly panel.

    Args:
        panel: clean panel with month_start (datetime) and raw columns.
        drop_incomplete: drop rows whose 12-month history is not yet available.

    Returns:
        DataFrame with `target_month`, `origin_month`, ids, features, and target `y`.
    """
    df = panel.sort_values(GROUP + ["month_start"]).reset_index(drop=True).copy()
    g = df.groupby(GROUP, sort=False)

    out = pd.DataFrame({
        "target_month": df["month_start"],
        "origin_month": df["month_start"] - pd.DateOffset(months=1),
        "region_id": df["region_id"],
        "product_id": df["product_id"],
        TARGET: df["units_sold"].astype(float),
    })
    for k in LAGS:
        out[f"lag_{k}"] = g["units_sold"].shift(k)

    history = g["units_sold"].shift(1)  # value observed at the origin month
    hist_g = history.groupby([df["region_id"], df["product_id"]], sort=False)
    for w in WINDOWS:
        out[f"roll_mean_{w}"] = hist_g.transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
        out[f"roll_std_{w}"] = hist_g.transform(lambda s, w=w: s.rolling(w, min_periods=w).std())

    out["spend_origin"] = g["marketing_spend_inr"].shift(1)
    spend_hist = out["spend_origin"].groupby([df["region_id"], df["product_id"]], sort=False)
    out["spend_sum_3"] = spend_hist.transform(lambda s: s.rolling(3, min_periods=3).sum())
    out["price_origin"] = g["unit_price_inr"].shift(1)
    out["competitor_origin"] = g["competitor_index"].shift(1)
    out["accounts_origin"] = g["active_accounts"].shift(1)
    out["campaign_origin"] = g["campaign_flag"].shift(1).astype(float)

    month = out["target_month"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)

    if drop_incomplete:
        out = out.dropna(subset=NUMERIC_FEATURES).reset_index(drop=True)
    return out.sort_values(["target_month"] + GROUP).reset_index(drop=True)


def next_month_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """Feature rows for the month after the panel ends (target unknown, y = NaN).

    A placeholder month with missing realized values is appended per series; every feature
    still comes only from observed history, so the frame is a genuine out-of-sample input.
    """
    last = panel["month_start"].max()
    nxt = last + pd.DateOffset(months=1)
    stub = panel[panel["month_start"] == last][GROUP].copy()
    stub["month_start"] = nxt
    extended = pd.concat([panel, stub], ignore_index=True)
    feats = build_features(extended, drop_incomplete=False)
    feats = feats[feats["target_month"] == nxt].reset_index(drop=True)
    if feats[NUMERIC_FEATURES].isna().any().any():
        raise ValueError("next-month features incomplete; check panel history")
    return feats


def main() -> None:
    config.setup_logging()
    config.ensure_dirs()
    panel = pd.read_parquet(config.PANEL_PARQUET)
    feats = build_features(panel)
    feats.to_parquet(config.FEATURES_PARQUET, index=False)
    logger.info("Feature rows=%d, target months %s..%s", len(feats),
                feats["target_month"].min().date(), feats["target_month"].max().date())


if __name__ == "__main__":
    main()
