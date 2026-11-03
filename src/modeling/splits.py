"""Chronological splits on target month. No random shuffling anywhere."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import config


@dataclass(frozen=True)
class Fold:
    """One expanding-window fold: train on targets <= train_end, evaluate on [val_start, val_end]."""

    name: str
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def backtest_folds() -> list[Fold]:
    """Tuning folds inside 2023-2024; none touches the 2025 final test."""
    return [Fold(f"fold{i + 1}", f["train_end"], f["val_start"], f["val_end"])
            for i, f in enumerate(config.BACKTEST_FOLDS)]


def final_fold() -> Fold:
    """Refit on all targets through Dec 2024, test once on Jan-Dec 2025."""
    return Fold("test_2025", config.VALID_END, config.TEST_START, config.TEST_END)


def split(df: pd.DataFrame, fold: Fold, date_col: str = "target_month") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, eval) rows for a fold; asserts strict temporal ordering."""
    train = df[df[date_col] <= fold.train_end]
    evaluation = df[(df[date_col] >= fold.val_start) & (df[date_col] <= fold.val_end)]
    if len(train) and len(evaluation):
        assert train[date_col].max() < evaluation[date_col].min(), "train must precede evaluation"
    return train, evaluation


def train_valid_test(df: pd.DataFrame, date_col: str = "target_month") -> dict[str, pd.DataFrame]:
    """Named partitions: train (<= 2023), valid (2024), test (2025)."""
    d = df[date_col]
    return {
        "train": df[d <= config.TRAIN_END],
        "valid": df[(d >= config.VALID_START) & (d <= config.VALID_END)],
        "test": df[(d >= config.TEST_START) & (d <= config.TEST_END)],
    }
