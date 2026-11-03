"""Tune XGBoost on backtest folds, then fit the production model on all history.

Usage: python -m ml.train_xgboost

* Hyperparameters are chosen on the 2023-2024 folds only (2025 is never seen during tuning).
* Evaluation of this *procedure* happens in `ml/evaluate.py` with a model refit through Dec 2024.
* The production model served by the API is refit on every target month through Dec 2025 with
  the selected hyperparameters, so live forecasts use the latest observed history.
"""

from __future__ import annotations

import logging

import pandas as pd

from ml import bundle
from src import config
from src.features.build import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, build_features
from src.modeling import xgboost_model
from src.utils.io import save_model

logger = logging.getLogger("ml.train_xgboost")


def main() -> dict:
    config.setup_logging()
    panel = pd.read_parquet(config.PANEL_PARQUET)
    features = build_features(panel)

    params, tuning = xgboost_model.tune(features)
    bundle.METRICS.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(bundle.METRICS / "xgboost_tuning.csv", index=False)

    pipe = xgboost_model.make_pipeline(params).fit(features[FEATURES], features["y"])
    save_model(pipe, bundle.XGB_MODEL)
    digest = bundle.sha256(bundle.XGB_MODEL)[:10]
    meta = {
        "model_name": "xgboost",
        "model_version": f"xgb-{config.EQUATION_VERSION}-{digest}",
        "trained_at_utc": bundle.now_utc(),
        "data_version": config.DATA_SOURCE,
        "seed": config.RANDOM_SEED,
        "params": {**params, **xgboost_model.FIXED_PARAMS},
        "selection": "lowest mean WAPE over 3 expanding folds (validation Jul 2023 - Dec 2024)",
        "train_target_months": [str(features["target_month"].min().date()), str(features["target_month"].max().date())],
        "n_train_rows": int(len(features)),
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "target": "units_sold in the target month (one month after the forecast origin)",
        "packages": bundle.package_versions(bundle.RUNTIME_PACKAGES),
        "note": "Refit on all history after evaluation; reported metrics come from the Dec-2024-cutoff refit in ml/evaluate.py.",
    }
    bundle.write_json(meta, bundle.XGB_META)
    logger.info("Saved %s (%s), params=%s", bundle.XGB_MODEL.name, meta["model_version"], params)
    return meta


if __name__ == "__main__":
    main()
