import pandas as pd

from src import config
from src.modeling.splits import backtest_folds, final_fold, split, train_valid_test


def test_temporal_split_order(features):
    parts = train_valid_test(features)
    assert parts["train"]["target_month"].max() < parts["valid"]["target_month"].min()
    assert parts["valid"]["target_month"].max() < parts["test"]["target_month"].min()
    assert parts["test"]["target_month"].min() == pd.Timestamp("2025-01-01")
    assert sum(len(p) for p in parts.values()) == len(features)


def test_backtest_folds_never_touch_test_year(features):
    for fold in backtest_folds():
        train, val = split(features, fold)
        assert train["target_month"].max() < val["target_month"].min()
        assert val["target_month"].max() < config.TEST_START
    train, test = split(features, final_fold())
    assert train["target_month"].max() == config.VALID_END
    assert len(test) == 12 * 18
