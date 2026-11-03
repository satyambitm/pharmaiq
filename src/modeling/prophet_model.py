"""Univariate Prophet per (region, product) series, refit at every one-step-ahead origin.

Input schema required by Prophet: `ds` (datetime, month start) and `y` (float units).
For target month T, the model is fit only on months < T (history available at origin T-1),
then predicts the single timestamp T. No regressors are used, because their future values
would not be known at forecast time.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from src.modeling.naive import KEY_COLS

logger = logging.getLogger(__name__)

PROPHET_PARAMS = {
    "yearly_seasonality": True,
    "weekly_seasonality": False,
    "daily_seasonality": False,
    "seasonality_mode": "additive",
}


def _quiet_cmdstan() -> None:
    for name in ("cmdstanpy", "prophet"):
        logging.getLogger(name).setLevel(logging.WARNING)


def to_prophet_frame(series: pd.DataFrame) -> pd.DataFrame:
    """Convert one series (month_start, units_sold) into Prophet's ds/y schema."""
    df = pd.DataFrame({"ds": pd.to_datetime(series["month_start"]), "y": series["units_sold"].astype(float)})
    return df.sort_values("ds").reset_index(drop=True)


def forecast_one_step(history: pd.DataFrame, target_month: pd.Timestamp) -> float:
    """Fit Prophet on `history` (ds/y strictly before target) and predict `target_month`."""
    from prophet import Prophet

    if history["ds"].max() >= target_month:
        raise ValueError("Prophet history must end before the target month")
    m = Prophet(**PROPHET_PARAMS)
    m.fit(history)
    pred = m.predict(pd.DataFrame({"ds": [target_month]}))
    return float(max(0.0, pred["yhat"].iloc[0]))


def rolling_forecast(panel: pd.DataFrame, target_rows: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """One-step-ahead Prophet predictions for exactly the (target_month, region, product) rows given.

    Returns predictions in the same key schema as other models, plus total runtime in seconds.
    """
    _quiet_cmdstan()
    start = time.perf_counter()
    preds = []
    for (r, p), rows in target_rows.groupby(["region_id", "product_id"]):
        series = to_prophet_frame(panel[(panel["region_id"] == r) & (panel["product_id"] == p)])
        for _, row in rows.iterrows():
            hist = series[series["ds"] < row["target_month"]]
            preds.append({"target_month": row["target_month"], "region_id": r, "product_id": p,
                          "y": row["y"], "y_pred": forecast_one_step(hist, row["target_month"])})
        logger.info("Prophet done for %s/%s (%d origins)", r, p, len(rows))
    out = pd.DataFrame(preds)[KEY_COLS + ["y", "y_pred"]]
    out["model"] = "prophet"
    return out, time.perf_counter() - start
