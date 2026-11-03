"""Multi-horizon (1-3 months) chronological evaluation, interval calibration and SHAP export.

Usage: python -m ml.evaluate          (run ml.train_xgboost first; takes ~2-3 minutes for Prophet)

Contract
* Test origins O = Dec 2024 ... Nov 2025; targets O + h within 2025, for h = 1, 2, 3.
* XGBoost: hyperparameters from ml.train_xgboost, refit on targets <= Dec 2024 (before every
  test origin), forecasting recursively from each origin using only data <= O.
* Prophet: refit per series at every origin on history <= O; forecasts h = 1..3 directly.
* Seasonal naive: units in the same month one year earlier (known at every origin for h <= 12).
* Within each horizon all three models are scored on exactly the same (target, region, product) rows.
* XGBoost 80% intervals use relative-error quantiles from a separate 2024 calibration run
  (model fit on targets <= Dec 2023), so 2025 coverage is an honest out-of-sample check.
"""

from __future__ import annotations

import logging
import shutil
import time

import numpy as np
import pandas as pd

from ml import bundle
from ml.train_prophet import INTERVAL_WIDTH, fit_predict
from src import config
from src.features.build import FEATURES, build_features
from src.modeling import xgboost_model
from src.modeling.compare import assert_same_rows
from src.modeling.metrics import score
from src.modeling.prophet_model import _quiet_cmdstan, to_prophet_frame
from src.modeling.recursive import MAX_HORIZON, recursive_forecast

logger = logging.getLogger("ml.evaluate")
KEYS = ["target_month", "region_id", "product_id"]
Q_LOW, Q_HIGH = (1 - INTERVAL_WIDTH) / 2, 1 - (1 - INTERVAL_WIDTH) / 2


def origins(first: str, last: str) -> list[pd.Timestamp]:
    return list(pd.date_range(first, last, freq="MS"))


def xgb_recursive(panel, pipe, origin_list, last_target) -> pd.DataFrame:
    out = pd.concat([recursive_forecast(panel, pipe, o, MAX_HORIZON) for o in origin_list], ignore_index=True)
    return out[out["target_month"] <= last_target]


def prophet_backtest(panel, origin_list, last_target) -> tuple[pd.DataFrame, float]:
    _quiet_cmdstan()
    start, rows = time.perf_counter(), []
    for (r, p), g in panel.groupby(["region_id", "product_id"]):
        series = to_prophet_frame(g)
        for o in origin_list:
            fc = fit_predict(series[series["ds"] <= o], MAX_HORIZON)
            for step, rec in enumerate(fc.itertuples(index=False), start=1):
                rows.append({"origin_month": o, "target_month": rec.ds, "step": step, "region_id": r,
                             "product_id": p, "y_pred": rec.yhat, "lower": rec.yhat_lower, "upper": rec.yhat_upper})
        logger.info("Prophet backtest %s/%s done", r, p)
    out = pd.DataFrame(rows)
    return out[out["target_month"] <= last_target], time.perf_counter() - start


def residual_quantiles(panel, features, params) -> dict[int, dict[str, float]]:
    """Relative-error quantiles per horizon from 2024 targets with a model fit through Dec 2023."""
    train = features[features["target_month"] <= config.TRAIN_END]
    pipe = xgboost_model.make_pipeline(params).fit(train[FEATURES], train["y"])
    fc = xgb_recursive(panel, pipe, origins("2023-12-01", "2024-11-01"), config.VALID_END)
    fc = fc.merge(panel.rename(columns={"month_start": "target_month"})[KEYS + ["units_sold"]], on=KEYS)
    fc["rel_err"] = fc["units_sold"] / fc["y_pred"] - 1
    return {int(h): {"q_low": float(g["rel_err"].quantile(Q_LOW)), "q_high": float(g["rel_err"].quantile(Q_HIGH)),
                     "n": int(len(g))} for h, g in fc.groupby("step")}


def export_shap(pipe, features) -> None:
    from src.modeling.explain import compute_shap

    test = features[(features["target_month"] >= config.TEST_START) & (features["target_month"] <= config.TEST_END)].reset_index(drop=True)
    values, importance = compute_shap(pipe, test)
    bundle.write_json({
        "model": "xgboost (fit on targets <= 2024-12)", "rows_explained": int(len(test)),
        "rows": "2025 one-step-ahead holdout rows", "unit": "units sold",
        "caveat": "SHAP values describe this model's predictions (associations), not causal effects. "
                  "Correlated lag features can share or swap credit.",
        "importance": [{"feature": r.feature, "mean_abs_shap": round(float(r.mean_abs_shap), 3)} for r in importance.head(15).itertuples()],
    }, bundle.SHAP_GLOBAL)
    examples = []
    pick = test[(test["region_id"] == "R01") & (test["target_month"] == config.TEST_END)]
    for i in pick.index:
        v = values[i]
        order = np.argsort(-np.abs(v.values))[:8]
        examples.append({
            "region_id": test.loc[i, "region_id"], "product_id": test.loc[i, "product_id"],
            "target_month": str(test.loc[i, "target_month"].date()), "actual_units": float(test.loc[i, "y"]),
            "base_value": round(float(v.base_values), 2), "prediction": round(float(v.base_values + v.values.sum()), 2),
            "contributions": [{"feature": v.feature_names[j], "feature_value": round(float(v.data[j]), 4),
                               "shap": round(float(v.values[j]), 2)} for j in order],
            "other_features_shap": round(float(v.values.sum() - v.values[order].sum()), 2),
        })
    bundle.write_json(examples, bundle.SHAP_LOCAL)


def main() -> None:
    config.setup_logging()
    bundle.METRICS.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(config.PANEL_PARQUET)
    features = build_features(panel)
    params = {k: bundle.read_json(bundle.XGB_META)["params"][k] for k in xgboost_model.PARAM_GRID}

    quantiles = residual_quantiles(panel, features, params)
    bundle.write_json({"interval_width": INTERVAL_WIDTH, "method": "relative-error quantiles (actual/pred - 1) "
                       "from 2024 recursive backtest, model fit on targets <= 2023-12",
                       "per_horizon": {str(k): v for k, v in quantiles.items()}}, bundle.XGB_RESIDUALS)

    train = features[features["target_month"] <= config.VALID_END]
    t0 = time.perf_counter()
    eval_pipe = xgboost_model.make_pipeline(params).fit(train[FEATURES], train["y"])
    test_origins = origins("2024-12-01", "2025-11-01")
    xgb = xgb_recursive(panel, eval_pipe, test_origins, config.TEST_END).assign(model="xgboost")
    xgb_secs = time.perf_counter() - t0
    q = pd.DataFrame(quantiles).T.rename_axis("step").reset_index()[["step", "q_low", "q_high"]]
    xgb = xgb.merge(q, on="step")
    xgb["lower"], xgb["upper"] = xgb["y_pred"] * (1 + xgb["q_low"]), xgb["y_pred"] * (1 + xgb["q_high"])
    xgb = xgb.drop(columns=["q_low", "q_high"])

    prophet, prophet_secs = prophet_backtest(panel, test_origins, config.TEST_END)
    prophet["model"] = "prophet"

    actual = panel.rename(columns={"month_start": "target_month"})[KEYS + ["units_sold"]].rename(columns={"units_sold": "y"})
    last_year = panel.assign(target_month=panel["month_start"] + pd.DateOffset(months=12))[KEYS + ["units_sold"]]
    naive = xgb[["origin_month", "target_month", "step", "region_id", "product_id"]].merge(
        last_year.rename(columns={"units_sold": "y_pred"}), on=KEYS).assign(model="seasonal_naive")

    preds = pd.concat([xgb, prophet, naive], ignore_index=True).merge(actual, on=KEYS)
    preds.to_csv(bundle.METRICS / "horizon_predictions.csv", index=False)

    rows, region_rows, coverage = [], [], []
    for h in range(1, MAX_HORIZON + 1):
        per_model = [g.assign(model=m) for m, g in preds[preds["step"] == h].groupby("model")]
        assert_same_rows(per_model)
        for g in per_model:
            m = g["model"].iloc[0]
            rows.append({"model": m, "horizon_months": h, "origins": f"{g['origin_month'].min():%Y-%m}..{g['origin_month'].max():%Y-%m}",
                         "targets": f"{g['target_month'].min():%Y-%m}..{g['target_month'].max():%Y-%m}", **score(g),
                         "runtime_seconds_all_horizons": round({"xgboost": xgb_secs, "prophet": prophet_secs}.get(m, 0.0), 2)})
            for r, gr in g.groupby("region_id"):
                region_rows.append({"model": m, "horizon_months": h, "region_id": r, **score(gr)})
            if "lower" in g and g["lower"].notna().all():
                inside = (g["y"] >= g["lower"]) & (g["y"] <= g["upper"])
                coverage.append({"model": m, "horizon_months": h, "nominal": INTERVAL_WIDTH,
                                 "empirical_coverage": round(float(inside.mean()), 4), "n": int(len(g)),
                                 "mean_width_units": round(float((g["upper"] - g["lower"]).mean()), 1)})
    pd.DataFrame(rows).to_csv(bundle.HORIZON_METRICS, index=False)
    pd.DataFrame(region_rows).to_csv(bundle.HORIZON_REGION_METRICS, index=False)
    pd.DataFrame(coverage).to_csv(bundle.INTERVAL_COVERAGE, index=False)

    export_shap(eval_pipe, features)
    for src_name, dst in (("model_comparison.csv", bundle.ONE_STEP_SCORECARD),):
        src_path = config.METRICS_DIR / src_name
        if src_path.exists():
            shutil.copyfile(src_path, dst)
        else:
            logger.warning("%s not found; run scripts/train_all.py to include the one-step fold scorecard", src_name)

    logger.info("Horizon metrics:\n%s", pd.DataFrame(rows)[["model", "horizon_months", "n_predictions", "MAE_units", "RMSE_units", "WAPE_%"]].to_string(index=False))
    logger.info("Interval coverage:\n%s", pd.DataFrame(coverage).to_string(index=False))


if __name__ == "__main__":
    main()
