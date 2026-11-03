"""Stored evaluation results and SHAP explanations (computed offline by `ml/evaluate.py`)."""

from __future__ import annotations

import math

from ml.model_loader import ModelBundle

EVALUATION = ("Chronological backtest: forecast origins Dec 2024 - Nov 2025, targets within 2025, "
              "horizons 1-3 months; all models scored on identical rows per horizon. "
              "XGBoost was refit on targets up to Dec 2024 for this evaluation.")


def _clean(v):
    return None if isinstance(v, float) and math.isnan(v) else v


def model_insights(bundle: ModelBundle) -> dict:
    hm = bundle.horizon_metrics.sort_values(["horizon_months", "WAPE_%"])
    one_step = None
    if bundle.one_step_scorecard is not None:
        one_step = [{k: _clean(v) for k, v in r.items()} for r in bundle.one_step_scorecard.to_dict("records")]
    return {
        "evaluation": EVALUATION,
        "horizon_scorecard": [{"model": r["model"], "horizon_months": int(r["horizon_months"]),
                               "n_predictions": int(r["n_predictions"]), "mae_units": round(float(r["MAE_units"]), 1),
                               "rmse_units": round(float(r["RMSE_units"]), 1), "wape_pct": round(float(r["WAPE_%"]), 2)}
                              for _, r in hm.iterrows()],
        "one_step_scorecard": one_step,
        "interval_coverage": bundle.interval_coverage.to_dict("records"),
        "shap_caveat": bundle.shap_global["caveat"],
        "global_importance": bundle.shap_global["importance"],
        "local_examples": bundle.shap_local,
        "model_versions": {"xgboost": bundle.xgb_meta["model_version"], "prophet": bundle.prophet_meta["model_version"]},
    }
