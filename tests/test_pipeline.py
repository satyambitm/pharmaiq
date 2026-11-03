"""Recursive multi-month forecasting: consistency with the one-step model and no lookahead."""

import numpy as np
import pandas as pd
import pytest

from src.features.build import FEATURES
from src.modeling import xgboost_model
from src.modeling.recursive import recursive_forecast

ORIGIN = pd.Timestamp("2024-06-01")
PARAMS = {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 50}


@pytest.fixture(scope="module")
def pipe(features):
    train = features[features["target_month"] <= ORIGIN]
    return xgboost_model.make_pipeline(PARAMS).fit(train[FEATURES], train["y"])


def test_step1_equals_one_step_prediction(panel, features, pipe):
    rec = recursive_forecast(panel, pipe, ORIGIN, 1)
    direct = features[features["origin_month"] == ORIGIN].copy()
    direct["direct"] = pipe.predict(direct[FEATURES]).clip(min=0)
    merged = rec.merge(direct[["region_id", "product_id", "direct"]], on=["region_id", "product_id"])
    assert len(merged) == 18
    np.testing.assert_allclose(merged["y_pred"], merged["direct"], rtol=1e-6)


def test_future_rows_cannot_change_forecast(panel, pipe):
    base = recursive_forecast(panel, pipe, ORIGIN, 3)
    tampered = panel.copy()
    future = tampered["month_start"] > ORIGIN
    tampered.loc[future, "units_sold"] = 10**7
    tampered.loc[future, "marketing_spend_inr"] = 0.0
    pd.testing.assert_frame_equal(base, recursive_forecast(tampered, pipe, ORIGIN, 3))


def test_output_shape_and_months(panel, pipe):
    out = recursive_forecast(panel, pipe, ORIGIN, 3)
    assert len(out) == 18 * 3
    assert sorted(out["target_month"].unique()) == list(pd.date_range("2024-07-01", periods=3, freq="MS"))
    assert (out["y_pred"] >= 0).all()


@pytest.mark.parametrize("horizon", [0, 4])
def test_rejects_unsupported_horizon(panel, pipe, horizon):
    with pytest.raises(ValueError):
        recursive_forecast(panel, pipe, ORIGIN, horizon)


def test_rejects_short_history(panel, pipe):
    with pytest.raises(ValueError):
        recursive_forecast(panel, pipe, pd.Timestamp("2021-06-01"), 1)
