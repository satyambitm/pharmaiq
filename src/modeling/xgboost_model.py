"""XGBoost one-step-ahead model: one-hot ids + numeric history features, small time-aware grid."""

from __future__ import annotations

import itertools
import logging
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from src import config
from src.features.build import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET
from src.modeling.metrics import score
from src.modeling.naive import KEY_COLS
from src.modeling.splits import backtest_folds, split

logger = logging.getLogger(__name__)

PARAM_GRID: dict[str, list] = {
    "max_depth": [3, 5],
    "learning_rate": [0.03, 0.08],
    "n_estimators": [300, 700],
}
FIXED_PARAMS = {"subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "reg_lambda": 1.0}


def make_pipeline(params: dict, seed: int = config.RANDOM_SEED) -> Pipeline:
    """Preprocessing + regressor. n_jobs=1 keeps training bit-for-bit deterministic."""
    pre = ColumnTransformer(
        [("ids", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    model = XGBRegressor(objective="reg:squarederror", random_state=seed, n_jobs=1,
                         tree_method="hist", **FIXED_PARAMS, **params)
    return Pipeline([("pre", pre), ("model", model)])


def predict(pipe: Pipeline, rows: pd.DataFrame, model_name: str = "xgboost") -> pd.DataFrame:
    """Predict units for rows; negative predictions are clipped to zero (reported rule)."""
    out = rows[KEY_COLS + [TARGET]].copy()
    out["y_pred"] = np.clip(pipe.predict(rows[FEATURES]), 0, None)
    out["model"] = model_name
    return out


def param_candidates() -> list[dict]:
    """All grid combinations (8 configurations)."""
    keys = list(PARAM_GRID)
    return [dict(zip(keys, vals)) for vals in itertools.product(*PARAM_GRID.values())]


def tune(features: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Select params by mean WAPE over the expanding backtest folds (2025 is never seen)."""
    hist = features[features["target_month"] <= config.VALID_END]
    rows = []
    for params in param_candidates():
        fold_scores = []
        for fold in backtest_folds():
            train, val = split(hist, fold)
            pipe = make_pipeline(params).fit(train[FEATURES], train[TARGET])
            s = score(predict(pipe, val))
            fold_scores.append(s["WAPE_%"])
            rows.append({**params, "fold": fold.name, **s})
        logger.info("params=%s mean fold WAPE=%.3f", params, float(np.mean(fold_scores)))
    results = pd.DataFrame(rows)
    summary = results.groupby(list(PARAM_GRID), as_index=False)["WAPE_%"].mean()
    best = summary.sort_values(["WAPE_%"] + list(PARAM_GRID)).iloc[0]
    best_params = {k: (int(best[k]) if k != "learning_rate" else float(best[k])) for k in PARAM_GRID}
    return best_params, results


def fit_final(features: pd.DataFrame, params: dict) -> tuple[Pipeline, float]:
    """Refit on every target month through Dec 2024; returns (pipeline, seconds)."""
    train = features[features["target_month"] <= config.VALID_END]
    start = time.perf_counter()
    pipe = make_pipeline(params).fit(train[FEATURES], train[TARGET])
    return pipe, time.perf_counter() - start


def feature_names(pipe: Pipeline) -> list[str]:
    """Column names after preprocessing (one-hot ids + numeric features)."""
    return list(pipe.named_steps["pre"].get_feature_names_out())


__all__ = ["make_pipeline", "predict", "tune", "fit_final", "feature_names", "NUMERIC_FEATURES"]
