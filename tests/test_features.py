import numpy as np
import pandas as pd

from src.features.build import NUMERIC_FEATURES, build_features, next_month_rows


def test_lag_feature_no_lookahead(panel, features):
    cutoff = pd.Timestamp("2024-06-01")
    shocked = panel.copy()
    shocked.loc[shocked["month_start"] >= cutoff, "units_sold"] *= 100
    shocked.loc[shocked["month_start"] >= cutoff, "marketing_spend_inr"] *= 100
    shocked.loc[shocked["month_start"] >= cutoff, "competitor_index"] += 5
    f2 = build_features(shocked)
    cols = ["target_month", "region_id", "product_id"] + NUMERIC_FEATURES
    before = features[features["target_month"] <= cutoff][cols].reset_index(drop=True)
    after = f2[f2["target_month"] <= cutoff][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after)


def test_lag_and_rolling_values_hand_checked(panel, features):
    s = panel[(panel["region_id"] == "R03") & (panel["product_id"] == "P02")].set_index("month_start")["units_sold"]
    row = features[(features["region_id"] == "R03") & (features["product_id"] == "P02")
                   & (features["target_month"] == pd.Timestamp("2023-05-01"))].iloc[0]
    assert row["y"] == s[pd.Timestamp("2023-05-01")]
    assert row["lag_1"] == s[pd.Timestamp("2023-04-01")]
    assert row["lag_12"] == s[pd.Timestamp("2022-05-01")]
    window = s[pd.Timestamp("2023-02-01"):pd.Timestamp("2023-04-01")]
    assert np.isclose(row["roll_mean_3"], window.mean())
    assert np.isclose(row["roll_std_3"], window.std())


def test_features_start_after_12_months_of_history(features):
    assert features["target_month"].min() == pd.Timestamp("2022-01-01")
    assert len(features) == 48 * 18
    assert not features[NUMERIC_FEATURES].isna().any().any()


def test_next_month_rows_use_history_only(panel):
    rows = next_month_rows(panel)
    assert len(rows) == 18
    assert (rows["target_month"] == pd.Timestamp("2026-01-01")).all()
    assert rows["y"].isna().all()
