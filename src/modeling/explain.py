"""SHAP explanations for the final XGBoost model on held-out 2025 rows.

SHAP values describe how features moved *this model's* predictions relative to its average
prediction. They are associations learned from synthetic data, not causal effects; correlated
lags can split or swap importance. No test targets are used to compute them.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src import config
from src.features.build import FEATURES
from src.modeling.xgboost_model import feature_names

logger = logging.getLogger(__name__)


def compute_shap(pipe: Pipeline, rows: pd.DataFrame) -> tuple[shap.Explanation, pd.DataFrame]:
    """SHAP values for `rows` (features only) plus a mean-|SHAP| global importance table."""
    X = pipe.named_steps["pre"].transform(rows[FEATURES])
    names = feature_names(pipe)
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    values = explainer(pd.DataFrame(X, columns=names))
    importance = (pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values.values).mean(axis=0)})
                  .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    return values, importance


def save_shap_artifacts(pipe: Pipeline, test_rows: pd.DataFrame, example_index: int = 0) -> pd.DataFrame:
    """Write global bar chart, one local waterfall, and the importance CSV; return importance."""
    values, importance = compute_shap(pipe, test_rows)
    importance.to_csv(config.METRICS_DIR / "shap_importance.csv", index=False)

    top = importance.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#2E86AB")
    ax.set_xlabel("Mean |SHAP value| (units)")
    ax.set_title("XGBoost global feature association - 2025 holdout (synthetic; not causal)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "shap_summary.png", dpi=120)
    plt.close(fig)

    row = test_rows.iloc[example_index]
    shap.plots.waterfall(values[example_index], max_display=10, show=False)
    plt.title(f"Local explanation: {row['region_id']}/{row['product_id']} target {row['target_month']:%Y-%m}",
              fontsize=10, pad=28)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "shap_waterfall_example.png", dpi=120)
    plt.close("all")
    logger.info("SHAP top features: %s", ", ".join(importance["feature"].head(5)))
    return importance
