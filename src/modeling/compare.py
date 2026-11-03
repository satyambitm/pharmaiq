"""Apples-to-apples scorecard: same grain, same horizon, identical prediction rows."""

from __future__ import annotations

import pandas as pd

from src.modeling.metrics import score, score_by
from src.modeling.naive import KEY_COLS
from src.modeling.splits import Fold

SCOPE = "all 18 region x product series"
HORIZON = "1 month ahead"


def assert_same_rows(predictions: list[pd.DataFrame]) -> None:
    """Every model must predict exactly the same unique (target_month, region, product) keys."""
    ref = None
    for p in predictions:
        if p.duplicated(KEY_COLS).any():
            raise AssertionError(f"{p['model'].iloc[0]} has duplicate prediction keys")
        keys = set(map(tuple, p[KEY_COLS].astype(str).to_numpy()))
        if ref is None:
            ref = keys
        elif keys != ref:
            raise AssertionError(f"{p['model'].iloc[0]} scored on different rows than the reference model")


def scorecard(predictions: list[pd.DataFrame], fold: Fold, runtimes: dict[str, float]) -> pd.DataFrame:
    """One row per model with the fair-comparison columns from the build contract."""
    assert_same_rows(predictions)
    rows = []
    for p in predictions:
        name = p["model"].iloc[0]
        rows.append({
            "model": name, "evaluation": fold.name, "scope": SCOPE, "horizon": HORIZON,
            "train_cutoff": fold.train_end.date().isoformat(),
            "test_start": fold.val_start.date().isoformat(), "test_end": fold.val_end.date().isoformat(),
            **score(p), "runtime_seconds": round(float(runtimes.get(name, float("nan"))), 3),
        })
    return pd.DataFrame(rows)


def per_region(predictions: list[pd.DataFrame], fold: Fold) -> pd.DataFrame:
    """Per-model, per-region metrics so large regions cannot hide weak small-region accuracy."""
    df = pd.concat(predictions, ignore_index=True)
    out = score_by(df, ["model", "region_id"])
    out.insert(1, "evaluation", fold.name)
    return out
