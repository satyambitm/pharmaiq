"""Recursive multi-month XGBoost forecasting, shared by offline evaluation and the API.

From forecast origin O (last observed month), step h predicts month O + h:
  * step 1 uses only observed history, identical to the one-step-ahead model;
  * step h > 1 feeds earlier *predicted* units back in as lag values;
  * exogenous inputs (spend, price, competitor index, accounts, campaign flag) are unknown for
    future months and are held at their origin-month values (persistence assumption).
Features always come from `build_features`, so inference uses exactly the training logic.
"""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from src.features.build import FEATURES, GROUP, build_features

EXOGENOUS = ["marketing_spend_inr", "unit_price_inr", "competitor_index", "active_accounts", "campaign_flag"]
PANEL_COLUMNS = ["month_start", "region_id", "product_id", "units_sold"] + EXOGENOUS
MAX_HORIZON = 3


def recursive_forecast(panel: pd.DataFrame, pipe: Pipeline, origin: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """Forecast `horizon` months after `origin` for every series in `panel`.

    Rows after `origin` are ignored, so the result cannot depend on future data.
    Returns columns: origin_month, target_month, step, region_id, product_id, y_pred.
    """
    if not 1 <= horizon <= MAX_HORIZON:
        raise ValueError(f"horizon must be between 1 and {MAX_HORIZON}")
    history = panel.loc[panel["month_start"] <= origin, PANEL_COLUMNS].copy()
    if history.empty:
        raise ValueError("no history at or before the forecast origin")
    held = history[history["month_start"] == origin].set_index(GROUP)[EXOGENOUS]

    outputs = []
    for step in range(1, horizon + 1):
        target = origin + pd.DateOffset(months=step)
        stub = held.reset_index()[GROUP].assign(month_start=target)
        feats = build_features(pd.concat([history, stub], ignore_index=True), drop_incomplete=False)
        rows = feats[feats["target_month"] == target].reset_index(drop=True)
        if rows[FEATURES].isna().any().any():
            raise ValueError("insufficient history to build features (need at least 13 months per series)")
        preds = pipe.predict(rows[FEATURES]).clip(min=0)
        step_out = rows[GROUP].assign(origin_month=origin, target_month=target, step=step, y_pred=preds)
        outputs.append(step_out)

        filled = step_out[GROUP].assign(month_start=target, units_sold=preds)
        filled = filled.merge(held.reset_index(), on=GROUP, how="left")
        history = pd.concat([history, filled[PANEL_COLUMNS]], ignore_index=True)

    return pd.concat(outputs, ignore_index=True)[["origin_month", "target_month", "step", "region_id", "product_id", "y_pred"]]
