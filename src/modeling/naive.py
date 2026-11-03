"""Baselines: seasonal naive (same month last year) and last-month naive."""

from __future__ import annotations

import pandas as pd

KEY_COLS = ["target_month", "region_id", "product_id"]


def seasonal_naive(rows: pd.DataFrame) -> pd.DataFrame:
    """Predict y(T) = units(T - 12), i.e. the `lag_12` feature already computed from history."""
    out = rows[KEY_COLS + ["y"]].copy()
    out["y_pred"] = rows["lag_12"].to_numpy(float)
    out["model"] = "seasonal_naive"
    return out


def last_month_naive(rows: pd.DataFrame) -> pd.DataFrame:
    """Predict y(T) = units(T - 1), the last observed month at the origin."""
    out = rows[KEY_COLS + ["y"]].copy()
    out["y_pred"] = rows["lag_1"].to_numpy(float)
    out["model"] = "last_month_naive"
    return out
