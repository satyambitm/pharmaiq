"""Optional per-series Prophet, evaluated on the same holdout months as XGBoost.

Prophet is a training-only dependency (root requirements.txt); when it is not installed in the worker
environment, the step is reported as skipped rather than silently replaced.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.training.forecast import KEY, score

MAX_SERIES = 20
MIN_MONTHS = 24


def available() -> bool:
    try:
        import prophet  # noqa: F401
    except Exception:  # ImportError, or a broken cmdstan install
        return False
    return True


def run(panel: pd.DataFrame, holdout: int, horizon: int, seed: int) -> dict:
    if not available():
        return {"status": "skipped", "reason": "Prophet is not installed in this training environment."}
    from prophet import Prophet

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)
    observed = panel[panel["units_sold"].notna()]
    counts = observed.groupby(KEY).size()
    last = panel["month_start"].max()
    live = set(map(tuple, observed[observed["month_start"] == last][KEY].to_numpy()))
    eligible = [k for k, n in counts.sort_values(ascending=False).items() if n >= MIN_MONTHS and k in live][:MAX_SERIES]
    if not eligible:
        return {"status": "skipped", "reason": f"No series with {MIN_MONTHS}+ observed months ending in the final month."}
    cutoff = last - pd.DateOffset(months=holdout)
    y_true, y_pred, forecasts = [], [], []
    for key in eligible:
        s = observed[(observed["region"] == key[0]) & (observed["product"] == key[1])][["month_start", "units_sold"]]
        s = s.rename(columns={"month_start": "ds", "units_sold": "y"})
        train, test = s[s["ds"] <= cutoff], s[s["ds"] > cutoff]
        if len(train) >= MIN_MONTHS - holdout and not test.empty:
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False).fit(train, seed=seed)
            pred = m.predict(test[["ds"]])["yhat"].clip(lower=0).to_numpy()
            y_true.extend(test["y"].tolist())
            y_pred.extend(pred.tolist())
        full = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False).fit(s, seed=seed)
        future = pd.DataFrame({"ds": [last + pd.DateOffset(months=h) for h in range(1, horizon + 1)]})
        fc = full.predict(future)
        for h, row in enumerate(fc.itertuples(), start=1):
            forecasts.append({"region": key[0], "product": key[1], "target_month": str(row.ds.date()), "step": h,
                              "y_pred": round(max(float(row.yhat), 0.0), 4),
                              "y_lower_80": round(max(float(row.yhat_lower), 0.0), 4),
                              "y_upper_80": round(max(float(row.yhat_upper), 0.0), 4)})
    return {
        "status": "ok", "series": [list(k) for k in eligible],
        "metrics": score(pd.Series(y_true, dtype=float), pd.Series(y_pred, dtype=float)),
        "evaluation": f"multi-month forecast from {cutoff.date()} over the {holdout} held-out months, top {len(eligible)} series by history",
        "forecasts": forecasts,
        "note": "Prophet 80% intervals come from Prophet's own uncertainty model.",
    }
