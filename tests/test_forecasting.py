import numpy as np
import pandas as pd
import pytest

from src.features.build import FEATURES
from src.modeling import compare, naive, xgboost_model
from src.modeling.metrics import mae, rmse, wape
from src.modeling.prophet_model import to_prophet_frame
from src.modeling.splits import final_fold, split


def test_metrics_definitions():
    a, p = np.array([10.0, 20.0, 30.0]), np.array([12.0, 18.0, 33.0])
    assert mae(a, p) == pytest.approx(7 / 3)
    assert rmse(a, p) == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
    assert wape(a, p) == pytest.approx(7 / 60 * 100)
    assert np.isnan(wape(np.zeros(3), p))


def test_model_prediction_row_key_uniqueness(features):
    train, test = split(features, final_fold())
    pipe = xgboost_model.make_pipeline({"max_depth": 3, "learning_rate": 0.1, "n_estimators": 50}).fit(train[FEATURES], train["y"])
    xgb = xgboost_model.predict(pipe, test)
    sn = naive.seasonal_naive(test)
    compare.assert_same_rows([sn, xgb])
    assert not xgb.duplicated(naive.KEY_COLS).any()
    assert np.isfinite(xgb["y_pred"]).all() and (xgb["y_pred"] >= 0).all()
    with pytest.raises(AssertionError):
        compare.assert_same_rows([sn, xgb.iloc[1:]])


def test_seasonal_naive_is_same_month_last_year(panel, features):
    _, test = split(features, final_fold())
    sn = naive.seasonal_naive(test).iloc[0]
    actual_ly = panel[(panel["region_id"] == sn["region_id"]) & (panel["product_id"] == sn["product_id"])
                      & (panel["month_start"] == sn["target_month"] - pd.DateOffset(months=12))]["units_sold"].iloc[0]
    assert sn["y_pred"] == actual_ly


def test_prophet_schema(panel):
    series = panel[(panel["region_id"] == "R01") & (panel["product_id"] == "P01")]
    df = to_prophet_frame(series)
    assert list(df.columns) == ["ds", "y"]
    assert pd.api.types.is_datetime64_any_dtype(df["ds"])
    assert pd.api.types.is_float_dtype(df["y"])
    assert df["ds"].is_monotonic_increasing and len(df) == 60
