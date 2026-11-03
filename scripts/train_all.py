"""Deterministic train + evaluate pipeline for all forecasting models.

Usage: python scripts/train_all.py [--skip-prophet]

Steps: build features -> tune XGBoost on 3 expanding folds (<= 2024) -> score seasonal naive,
last-month naive, XGBoost and Prophet on identical rows for each fold and for the untouched
2025 test -> save model, predictions, scorecards, SHAP, figures and run metadata.
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.features.build import FEATURES, build_features, next_month_rows  # noqa: E402
from src.modeling import compare, naive, prophet_model, xgboost_model  # noqa: E402
from src.modeling.explain import save_shap_artifacts  # noqa: E402
from src.modeling.splits import backtest_folds, final_fold, split  # noqa: E402
from src.utils.io import load_model, save_json, save_model  # noqa: E402

logger = logging.getLogger("train_all")
PACKAGES = ["numpy", "pandas", "scikit-learn", "xgboost", "prophet", "shap", "scipy", "streamlit"]


def evaluate_fold(panel, features, fold, params, run_prophet):
    """Fit/predict every model for one fold on the same evaluation rows."""
    train, evaluation = split(features, fold)
    preds = [naive.seasonal_naive(evaluation), naive.last_month_naive(evaluation)]
    runtimes = {"seasonal_naive": 0.0, "last_month_naive": 0.0}

    start = pd.Timestamp.now()
    pipe = xgboost_model.make_pipeline(params).fit(train[FEATURES], train["y"])
    preds.append(xgboost_model.predict(pipe, evaluation))
    runtimes["xgboost"] = (pd.Timestamp.now() - start).total_seconds()

    if run_prophet:
        p, secs = prophet_model.rolling_forecast(panel, evaluation)
        preds.append(p)
        runtimes["prophet"] = secs
    return pipe, preds, runtimes


def plot_forecasts(preds: pd.DataFrame) -> None:
    """Aggregate actual vs predicted units by month and one example series, 2025 holdout."""
    agg = preds.groupby(["model", "target_month"])[["y", "y_pred"]].sum().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    actual = agg[agg["model"] == agg["model"].iloc[0]]
    axes[0].plot(actual["target_month"], actual["y"], "k-o", lw=2, label="actual")
    for m, g in agg.groupby("model"):
        axes[0].plot(g["target_month"], g["y_pred"], "--", marker=".", label=m)
    axes[0].set(title="2025 holdout: total units, one-month-ahead", xlabel="Target month", ylabel="Units")
    axes[0].legend(fontsize=8)

    ex = preds[(preds["region_id"] == "R01") & (preds["product_id"] == "P02")]
    first = ex[ex["model"] == ex["model"].iloc[0]]
    axes[1].plot(first["target_month"], first["y"], "k-o", lw=2, label="actual")
    for m, g in ex.groupby("model"):
        axes[1].plot(g["target_month"], g["y_pred"], "--", marker=".", label=m)
    axes[1].set(title="Example series North / Therapy B", xlabel="Target month", ylabel="Units")
    axes[1].legend(fontsize=8)
    fig.text(0.99, 0.005, "Synthetic demo data", ha="right", fontsize=7, color="grey")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "forecast_comparison.png", dpi=120)
    plt.close(fig)


def next_month_forecast(panel: pd.DataFrame, pipe) -> pd.DataFrame:
    """Illustrative Jan 2026 forecast with an explicitly assumed price and discount."""
    rows = next_month_rows(panel)
    out = xgboost_model.predict(pipe, rows)
    last = panel[panel["month_start"] > panel["month_start"].max() - pd.DateOffset(months=12)]
    disc = (last.groupby(["region_id", "product_id"])[["discount_inr", "gross_revenue_inr"]].sum()
            .pipe(lambda d: d["discount_inr"] / d["gross_revenue_inr"]).rename("assumed_discount_rate").reset_index())
    out = out.merge(rows[["region_id", "product_id", "price_origin"]], on=["region_id", "product_id"])
    out = out.merge(disc, on=["region_id", "product_id"])
    out = out.rename(columns={"price_origin": "assumed_unit_price_inr"})
    out["estimated_net_revenue_inr"] = out["y_pred"] * out["assumed_unit_price_inr"] * (1 - out["assumed_discount_rate"])
    out["note"] = ("ESTIMATE: units from XGBoost; price = last observed (Dec 2025) realized price; "
                   "discount = trailing 12-month rate. Ignores the January list-price increase pattern.")
    return out.drop(columns=["y"])


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            versions[p] = importlib_metadata.version(p)
        except importlib_metadata.PackageNotFoundError:
            versions[p] = "not installed"
    return versions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-prophet", action="store_true", help="faster run without Prophet")
    args = parser.parse_args()
    config.setup_logging()
    config.ensure_dirs()
    run_prophet = not args.skip_prophet

    panel = pd.read_parquet(config.PANEL_PARQUET)
    features = build_features(panel)
    features.to_parquet(config.FEATURES_PARQUET, index=False)
    logger.info("Features: %d rows, targets %s..%s", len(features),
                features["target_month"].min().date(), features["target_month"].max().date())

    best_params, tuning = xgboost_model.tune(features)
    tuning.to_csv(config.METRICS_DIR / "xgboost_tuning.csv", index=False)
    logger.info("Selected XGBoost params (folds only, 2025 unseen): %s", best_params)

    cards, regions, all_fold_preds = [], [], []
    for fold in backtest_folds():
        _, preds, runtimes = evaluate_fold(panel, features, fold, best_params, run_prophet)
        cards.append(compare.scorecard(preds, fold, runtimes))
        regions.append(compare.per_region(preds, fold))
        all_fold_preds.extend(p.assign(evaluation=fold.name) for p in preds)

    test = final_fold()
    pipe, test_preds, runtimes = evaluate_fold(panel, features, test, best_params, run_prophet)
    cards.append(compare.scorecard(test_preds, test, runtimes))
    regions.append(compare.per_region(test_preds, test))

    scorecard = pd.concat(cards, ignore_index=True)
    scorecard.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)
    pd.concat(regions, ignore_index=True).to_csv(config.METRICS_DIR / "metrics_by_region.csv", index=False)
    holdout = pd.concat(test_preds, ignore_index=True)
    holdout.to_parquet(config.FORECASTS_DIR / "holdout_predictions.parquet", index=False)
    pd.concat(all_fold_preds, ignore_index=True).to_parquet(config.FORECASTS_DIR / "backtest_predictions.parquet", index=False)
    if run_prophet:
        holdout[holdout["model"] == "prophet"].to_parquet(config.FORECASTS_DIR / "prophet_predictions.parquet", index=False)

    model_path = config.MODELS_DIR / "xgboost.joblib"
    save_model(pipe, model_path)
    _, test_rows = split(features, test)
    reloaded = load_model(model_path)
    xgb_test = holdout[holdout["model"] == "xgboost"]["y_pred"].to_numpy()
    if not np.allclose(xgboost_model.predict(reloaded, test_rows)["y_pred"].to_numpy(), xgb_test):
        raise AssertionError("reloaded model predictions differ from saved predictions")

    save_shap_artifacts(pipe, test_rows.reset_index(drop=True))
    plot_forecasts(holdout)
    next_month_forecast(panel, pipe).to_csv(config.FORECASTS_DIR / "next_month_forecast.csv", index=False)

    save_json({
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_version": config.DATA_SOURCE, "seed": config.RANDOM_SEED,
        "train_end": str(config.TRAIN_END.date()), "valid": [str(config.VALID_START.date()), str(config.VALID_END.date())],
        "test": [str(config.TEST_START.date()), str(config.TEST_END.date())],
        "final_model_trained_on_targets_through": str(config.VALID_END.date()),
        "backtest_folds": [{"name": f.name, "train_end": str(f.train_end.date()), "val": [str(f.val_start.date()), str(f.val_end.date())]} for f in backtest_folds()],
        "xgboost_best_params": best_params, "prophet_params": prophet_model.PROPHET_PARAMS if run_prophet else None,
        "prophet_run": run_prophet, "feature_rows": len(features), "features": FEATURES,
        "packages": package_versions(),
    }, config.METRICS_DIR / "run_metadata.json")

    test_card = scorecard[scorecard["evaluation"] == test.name]
    logger.info("2025 holdout scorecard:\n%s", test_card[["model", "n_predictions", "MAE_units", "RMSE_units", "WAPE_%", "runtime_seconds"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
