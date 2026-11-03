"""Fit Prophet per series on all history and precompute 1-3 month forecasts with intervals.

Usage: python -m ml.train_prophet

Prophet is univariate here: it extrapolates each series' own trend and yearly seasonality and
uses no future-known features. Forecasts are precomputed because fitting Prophet on a small
cloud instance per request would be slow; the API serves this file and checks its origin.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from ml import bundle
from src import config
from src.modeling.prophet_model import PROPHET_PARAMS, _quiet_cmdstan, to_prophet_frame
from src.modeling.recursive import MAX_HORIZON

logger = logging.getLogger("ml.train_prophet")
INTERVAL_WIDTH = 0.8


def fit_predict(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Fit on ds/y history and return the next `horizon` monthly forecasts with intervals."""
    from prophet import Prophet

    m = Prophet(**PROPHET_PARAMS, interval_width=INTERVAL_WIDTH)
    m.fit(history)
    future = pd.DataFrame({"ds": pd.date_range(history["ds"].max() + pd.DateOffset(months=1), periods=horizon, freq="MS")})
    fc = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    for c in ("yhat", "yhat_lower", "yhat_upper"):
        fc[c] = fc[c].clip(lower=0)
    return fc


def main() -> None:
    config.setup_logging()
    _quiet_cmdstan()
    panel = pd.read_parquet(config.PANEL_PARQUET)
    origin = panel["month_start"].max()
    start = time.perf_counter()
    rows = []
    for (r, p), g in panel.groupby(["region_id", "product_id"]):
        fc = fit_predict(to_prophet_frame(g), MAX_HORIZON)
        for step, rec in enumerate(fc.itertuples(index=False), start=1):
            rows.append({"origin_month": origin.date(), "target_month": rec.ds.date(), "step": step,
                         "region_id": r, "product_id": p, "y_pred": rec.yhat,
                         "lower": rec.yhat_lower, "upper": rec.yhat_upper})
    out = pd.DataFrame(rows)
    out.to_csv(bundle.PROPHET_FORECASTS, index=False)
    bundle.write_json({
        "model_name": "prophet",
        "model_version": f"prophet-{config.EQUATION_VERSION}-origin{origin:%Y%m}",
        "trained_at_utc": bundle.now_utc(),
        "forecast_origin": str(origin.date()),
        "horizon_months": MAX_HORIZON,
        "params": PROPHET_PARAMS,
        "interval_width": INTERVAL_WIDTH,
        "interval_method": "Prophet uncertainty interval (trend + observation noise simulation)",
        "fit_seconds_total": round(time.perf_counter() - start, 2),
        "n_series": int(out[["region_id", "product_id"]].drop_duplicates().shape[0]),
        "packages": bundle.package_versions(["prophet", "pandas", "numpy"]),
    }, bundle.PROPHET_META)
    logger.info("Saved %d Prophet forecast rows from origin %s", len(out), origin.date())


if __name__ == "__main__":
    main()
