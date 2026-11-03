"""Forecast error metrics in units.

WAPE is volume-weighted: high-volume series dominate it, so per-region metrics are also reported.
MAPE is intentionally omitted from the scorecard because it explodes near zero actuals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(pred))))


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    """Root mean squared error (penalizes large misses more than MAE)."""
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(pred)) ** 2)))


def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    """Weighted absolute percentage error in %; NaN if total actual volume is zero."""
    denom = float(np.sum(np.abs(actual)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(np.abs(np.asarray(actual) - np.asarray(pred))) / denom * 100)


def score(df: pd.DataFrame, actual_col: str = "y", pred_col: str = "y_pred") -> dict[str, float]:
    """MAE/RMSE/WAPE and row count for a prediction frame."""
    a, p = df[actual_col].to_numpy(float), df[pred_col].to_numpy(float)
    return {"n_predictions": int(len(df)), "MAE_units": mae(a, p), "RMSE_units": rmse(a, p), "WAPE_%": wape(a, p)}


def score_by(df: pd.DataFrame, by: list[str], actual_col: str = "y", pred_col: str = "y_pred") -> pd.DataFrame:
    """Metrics per group (e.g. model x region)."""
    rows = []
    for keys, grp in df.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append({**dict(zip(by, keys)), **score(grp, actual_col, pred_col)})
    return pd.DataFrame(rows)
