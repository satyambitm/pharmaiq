"""Train every model a dataset supports and return JSON-safe results plus a binary artifact.

The API serves the stored results JSON and never unpickles the artifact, which is kept only for audit and
reproduction offline. Every number in the results comes from an actual fit or evaluation; models the data
cannot support are reported as skipped with the reason.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata

import joblib
import numpy as np
import pandas as pd

from src.training import TRAINING_VERSION, prophet_series, segments
from src.training.forecast import (HORIZON, MIN_MEDIAN_MONTHS, MIN_TRAIN_ROWS, choose_spec, evaluate, fit_final,
                                   naive_forecast, recursive_forecast, regular_panel, shap_importance)

SEED = 42
PACKAGES = ["xgboost", "xgboost-cpu", "scikit-learn", "pandas", "numpy", "prophet"]
SHAP_CAVEAT = ("SHAP values show how much each input moved this model's predictions on past data. They describe "
               "associations learned by the model, not causal effects, and do not measure marketing ROI.")


@dataclass
class TrainingOutput:
    results: dict
    artifact: bytes


def _versions() -> dict:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = importlib_metadata.version(p)
        except importlib_metadata.PackageNotFoundError:
            continue
    return out


def _records(frame: pd.DataFrame, model: str) -> list[dict]:
    f = frame.assign(target_month=frame["target_month"].dt.strftime("%Y-%m-%d"), model=model,
                     y_pred=frame["y_pred"].astype(float).round(4))
    return f.to_dict("records")


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))


def train_dataset(rows: pd.DataFrame, report: dict, seed: int = SEED, with_prophet: bool = True) -> TrainingOutput:
    start = time.perf_counter()
    rows = rows.copy()
    rows["month_start"] = pd.to_datetime(rows["month_start"])
    panel = regular_panel(rows)
    observed_months = rows.groupby(["region", "product"]).size()
    median_months = float(observed_months.median())
    available = report.get("model_features", {}).get("available", [])
    spec = choose_spec(median_months, available)
    use_xgb = median_months >= MIN_MEDIAN_MONTHS

    fc: dict = {"status": "ok", "horizon_months": HORIZON, "median_months_per_series": median_months,
                "features": {"lags": list(spec.lags), "rolling_windows": list(spec.windows), "drivers": spec.drivers,
                             "drivers_not_used": [d for d in report.get("model_features", {}).get("not_mapped", [])]
                             + report.get("model_features", {}).get("too_sparse", [])},
                "notes": []}
    if not use_xgb:
        fc["notes"].append(f"Series have a median of {median_months:g} months; XGBoost needs {MIN_MEDIAN_MONTHS}+, "
                           "so only the naive baseline is provided.")
    if 12 not in spec.lags:
        fc["notes"].append("Less than 24 months per series: no lag-12 feature, so yearly seasonality is not learned.")
    fc["evaluation"] = evaluate(panel, spec, seed, use_xgb)
    metrics = fc["evaluation"]["metrics"]

    artifact: dict = {"training_version": TRAINING_VERSION, "spec": spec.__dict__}
    forecasts = _records(naive_forecast(panel), "naive_last_month")
    shap: dict = {"status": "skipped", "reason": "No XGBoost model was trained.", "caveat": SHAP_CAVEAT}
    if use_xgb and "xgboost" in metrics:
        pipe, train_rows = fit_final(panel, spec, seed)
        if len(train_rows) >= MIN_TRAIN_ROWS:
            xgb_fc, skipped = recursive_forecast(panel, spec, pipe)
            forecasts += _records(xgb_fc, "xgboost")
            if skipped:
                fc["notes"].append(f"{skipped} series lack the last three months and get only the naive forecast.")
            shap = {"status": "ok", "method": "TreeSHAP via XGBoost pred_contribs on up to 2,000 training rows",
                    "global_importance": shap_importance(pipe, train_rows, spec, seed), "caveat": SHAP_CAVEAT}
            artifact["xgboost_pipeline"] = pipe
            fc["n_final_training_rows"] = int(len(train_rows))
    elif use_xgb:
        fc["notes"].append(fc["evaluation"].get("xgboost_skipped", "XGBoost could not be evaluated, so it is not served."))

    xgb_wape = metrics.get("xgboost", {}).get("WAPE_pct")
    naive_wape = metrics.get("naive_last_month", {}).get("WAPE_pct")
    if xgb_wape is not None and naive_wape is not None and any(f["model"] == "xgboost" for f in forecasts):
        fc["recommended_model"] = "xgboost" if xgb_wape < naive_wape else "naive_last_month"
        if xgb_wape >= naive_wape:
            fc["notes"].append("XGBoost did not beat the naive forecast on the holdout; the naive forecast is recommended.")
    else:
        fc["recommended_model"] = "naive_last_month"
    fc["forecasts"] = forecasts
    fc["forecast_note"] = ("Point forecasts for the next months. Steps 2-3 feed predictions back as history and hold "
                           "drivers at their last values, so they are less reliable than step 1.")

    seg = segments.run(rows, seed)
    if seg["status"] == "ok":
        artifact["segments_k"] = seg["k"]

    prophet = ({"status": "skipped", "reason": "Disabled for this run."} if not with_prophet else
               prophet_series.run(panel, fc["evaluation"]["holdout_months"], HORIZON, seed))

    buf = io.BytesIO()
    joblib.dump(artifact, buf, compress=3)
    results = {
        "training_version": TRAINING_VERSION, "seed": seed,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {"monthly_rows": int(len(rows)), "series": int(len(observed_months)),
                 "date_start": str(rows["month_start"].min().date()), "date_end": str(rows["month_start"].max().date()),
                 "forecast_readiness": report.get("forecast_readiness", {}).get("level")},
        "forecasting": fc, "explainability": shap, "segmentation": seg, "prophet": prophet,
        "packages": _versions(), "duration_s": round(time.perf_counter() - start, 2),
    }
    return TrainingOutput(_jsonable(results), buf.getvalue())
