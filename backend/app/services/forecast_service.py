"""Region forecasts from the stored bundle. Models are never trained here.

XGBoost runs live (recursive, < 1 s) from the latest observed month; Prophet forecasts are
precomputed offline by `ml/train_prophet.py` because fitting Prophet per request is too slow
and too memory-hungry for a small instance.
"""

from __future__ import annotations

import threading

import pandas as pd

from backend.app.core.errors import InvalidRequest, ServiceUnavailable
from backend.app.db.connection import SalesRepository
from ml.model_loader import ModelBundle
from src.config import PRODUCTS, REGIONS
from src.modeling.recursive import MAX_HORIZON, recursive_forecast

HISTORY_MONTHS = 24
XGB_ASSUMPTIONS = [
    "Step 1 uses only observed history; steps 2-3 feed earlier predictions back in as lags.",
    "Future marketing spend, price, competitor index, accounts and campaign flag are unknown and "
    "are held at their latest observed values.",
    "Interval: 80% band from relative-error quantiles of a 2024 recursive backtest.",
]
PROPHET_ASSUMPTIONS = [
    "Fitted offline per region x product on all observed months; served as stored results.",
    "Uses only the units history (no spend, price or competitor inputs).",
    "Interval: Prophet's own 80% uncertainty interval.",
]

_cache: dict[tuple, pd.DataFrame] = {}
_lock = threading.Lock()


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def _coverage(bundle: ModelBundle, model: str) -> dict[int, float]:
    c = bundle.interval_coverage[bundle.interval_coverage["model"] == model]
    return {int(r.horizon_months): round(float(r.empirical_coverage), 4) for r in c.itertuples()}


def _accuracy(bundle: ModelBundle, model: str, horizon: int) -> list[dict]:
    m = bundle.horizon_metrics
    m = m[(m["model"] == model) & (m["horizon_months"] <= horizon)].sort_values("horizon_months")
    return [{"horizon_months": int(r["horizon_months"]), "mae_units": round(float(r["MAE_units"]), 1),
             "rmse_units": round(float(r["RMSE_units"]), 1), "wape_pct": round(float(r["WAPE_%"]), 2),
             "n_predictions": int(r["n_predictions"])} for _, r in m.iterrows()]


def _xgb(bundle: ModelBundle, region_panel: pd.DataFrame, region_id: str, origin: pd.Timestamp) -> pd.DataFrame:
    key = (region_id, origin, bundle.xgb_meta["model_version"])
    with _lock:
        if key in _cache:
            return _cache[key]
    fc = recursive_forecast(region_panel, bundle.xgb_pipeline, origin, MAX_HORIZON)
    q = bundle.xgb_residuals["per_horizon"]
    fc["lower"] = [p * (1 + q[str(s)]["q_low"]) for p, s in zip(fc["y_pred"], fc["step"])]
    fc["upper"] = [p * (1 + q[str(s)]["q_high"]) for p, s in zip(fc["y_pred"], fc["step"])]
    with _lock:
        _cache[key] = fc
    return fc


def forecast(repo: SalesRepository, bundle: ModelBundle, region_id: str, horizon: int, model: str) -> dict:
    if not 1 <= horizon <= MAX_HORIZON:
        raise InvalidRequest(f"horizon must be between 1 and {MAX_HORIZON} months")
    panel = repo.fetch_sales(region_ids=[region_id])
    if panel.empty:
        raise ServiceUnavailable("No sales history available for this region")
    origin = panel["month_start"].max()
    assumptions: list[str]

    if model == "xgboost":
        recent = panel[panel["month_start"] > origin - pd.DateOffset(months=HISTORY_MONTHS)]
        fc = _xgb(bundle, recent, region_id, origin)
        version, generated, assumptions = bundle.xgb_meta["model_version"], "live", list(XGB_ASSUMPTIONS)
        interval = {"level": bundle.xgb_residuals["interval_width"], "method": bundle.xgb_residuals["method"]}
    else:
        p = bundle.prophet_forecasts
        fc = p[p["region_id"] == region_id].copy()
        prophet_origin = pd.Timestamp(bundle.prophet_meta["forecast_origin"])
        version, generated, assumptions = bundle.prophet_meta["model_version"], "precomputed", list(PROPHET_ASSUMPTIONS)
        interval = {"level": bundle.prophet_meta["interval_width"], "method": bundle.prophet_meta["interval_method"]}
        if prophet_origin != origin:
            assumptions.append(f"Stored Prophet forecasts start after {prophet_origin:%b %Y}, but data now ends "
                               f"{origin:%b %Y}; rerun ml/train_prophet.py to refresh.")
        origin = prophet_origin

    fc = fc[fc["step"] <= horizon].sort_values(["product_id", "step"])
    history = panel[panel["month_start"] > origin - pd.DateOffset(months=HISTORY_MONTHS)]
    return {
        "region_id": region_id, "region_name": REGIONS[region_id], "model": model, "model_version": version,
        "origin_month": origin.date(), "horizon": horizon, "generated": generated, "assumptions": assumptions,
        "interval": {**interval, "empirical_coverage_2025": _coverage(bundle, model)},
        "forecast": [{"product_id": r.product_id, "product_name": PRODUCTS[r.product_id],
                      "target_month": pd.Timestamp(r.target_month).date(), "step": int(r.step),
                      "predicted_units": round(float(r.y_pred), 1), "lower_units": round(float(r.lower), 1),
                      "upper_units": round(float(r.upper), 1)} for r in fc.itertuples()],
        "history": [{"product_id": r.product_id, "month": r.month_start.date(), "units_sold": int(r.units_sold)}
                    for r in history.sort_values(["product_id", "month_start"]).itertuples()],
        "backtest_accuracy_2025": _accuracy(bundle, model, horizon),
    }
