"""One-month-ahead XGBoost for an uploaded monthly region x product panel, with honest baselines.

The feature logic mirrors `src/features/build.py` but adapts to what the dataset actually has:
  * lags 1-3 always; lag 6 and a 6-month window once series have 12+ months; lag 12 at 24+ months;
  * optional drivers (price, marketing, competitor, market) enter only if the quality report found them
    with >= 90 % monthly coverage, and only as origin-month values (shift 1), never same-month values;
  * gap months stay missing (no interpolation); rows without lags 1-3 are not used for training.

Evaluation holds out the last 3 (or 6, with 30+ months) calendar months and compares XGBoost with the
naive last-month forecast (and seasonal naive when lag 12 exists) on the same rows. Multi-month forecasts
are recursive: predicted units are fed back as lags and drivers are held at their last observed values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import DMatrix, XGBRegressor

KEY = ["region", "product"]
DRIVERS = ["unit_price", "marketing_spend", "doctor_interactions", "competitor_price", "competitor_count",
           "market_size", "hcp_count"]
XGB_PARAMS = {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8,
              "colsample_bytree": 0.8, "min_child_weight": 3, "reg_lambda": 1.0}
REQUIRED_LAGS = ["lag_1", "lag_2", "lag_3"]
MIN_TRAIN_ROWS = 40
MIN_MEDIAN_MONTHS = 12
HORIZON = 3
SHAP_SAMPLE = 2000


@dataclass
class FeatureSpec:
    lags: tuple[int, ...]
    windows: tuple[int, ...]
    drivers: list[str] = field(default_factory=list)

    @property
    def numeric(self) -> list[str]:
        return ([f"lag_{k}" for k in self.lags] + [f"roll_mean_{w}" for w in self.windows] + ["roll_std_3"]
                + [f"{d}_origin" for d in self.drivers] + ["month_sin", "month_cos"])

    @property
    def columns(self) -> list[str]:
        return KEY + self.numeric


def choose_spec(median_months: float, available_features: list[str]) -> FeatureSpec:
    lags = (1, 2, 3) + ((6,) if median_months >= 12 else ()) + ((12,) if median_months >= 24 else ())
    windows = (3,) + ((6,) if median_months >= 12 else ())
    return FeatureSpec(lags, windows, [d for d in DRIVERS if d in available_features])


def regular_panel(rows: pd.DataFrame) -> pd.DataFrame:
    """Every calendar month between each series' first and last month; gap months are all-missing rows."""
    parts = []
    for (region, product), g in rows.groupby(KEY, sort=True):
        idx = pd.date_range(g["month_start"].min(), g["month_start"].max(), freq="MS", name="month_start")
        part = g.drop(columns=KEY).set_index("month_start").reindex(idx).reset_index()
        parts.append(part.assign(region=region, product=product))
    return pd.concat(parts, ignore_index=True)[rows.columns]


def build_features(panel: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    df = panel.sort_values(KEY + ["month_start"]).reset_index(drop=True)
    units = df.groupby(KEY, sort=False)["units_sold"]
    out = df[KEY].copy()
    out["target_month"] = df["month_start"]
    out["y"] = df["units_sold"].astype(float)
    for k in spec.lags:
        out[f"lag_{k}"] = units.shift(k)
    hist = units.shift(1).groupby([df["region"], df["product"]], sort=False)
    for w in spec.windows:
        out[f"roll_mean_{w}"] = hist.transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
    out["roll_std_3"] = hist.transform(lambda s: s.rolling(3, min_periods=3).std())
    for d in spec.drivers:
        out[f"{d}_origin"] = df.groupby(KEY, sort=False)[d].shift(1)
    month = out["target_month"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    return out


def usable(features: pd.DataFrame) -> pd.DataFrame:
    return features[features["y"].notna() & features[REQUIRED_LAGS].notna().all(axis=1)].reset_index(drop=True)


def make_pipeline(spec: FeatureSpec, seed: int) -> Pipeline:
    pre = ColumnTransformer([("ids", OneHotEncoder(handle_unknown="ignore", sparse_output=False), KEY)],
                            remainder="passthrough", verbose_feature_names_out=False)
    model = XGBRegressor(objective="reg:squarederror", random_state=seed, n_jobs=1, tree_method="hist", **XGB_PARAMS)
    return Pipeline([("pre", pre), ("model", model)])


def score(y: pd.Series, pred: pd.Series) -> dict:
    """MAE, RMSE, WAPE and bias on the rows where both values exist."""
    ok = y.notna() & pred.notna()
    y, pred = y[ok].astype(float), pred[ok].astype(float)
    if y.empty:
        return {"n": 0, "MAE": None, "RMSE": None, "WAPE_pct": None, "bias_pct": None}
    err = pred - y
    total = float(y.abs().sum())
    return {"n": int(len(y)), "MAE": round(float(err.abs().mean()), 4), "RMSE": round(float(np.sqrt((err ** 2).mean())), 4),
            "WAPE_pct": round(float(err.abs().sum() / total * 100), 3) if total > 0 else None,
            "bias_pct": round(float(err.sum() / total * 100), 3) if total > 0 else None}


def holdout_months(months_spanned: int) -> int:
    return 6 if months_spanned >= 30 else 3


def evaluate(panel: pd.DataFrame, spec: FeatureSpec, seed: int, use_xgboost: bool) -> dict:
    """Fit on months before the holdout, score every model on the same held-out rows."""
    last = panel["month_start"].max()
    months = int((last.year - panel["month_start"].min().year) * 12 + last.month - panel["month_start"].min().month + 1)
    h = holdout_months(months)
    cutoff = last - pd.DateOffset(months=h)
    rows = usable(build_features(panel, spec))
    train, test = rows[rows["target_month"] <= cutoff], rows[rows["target_month"] > cutoff]
    out: dict = {"holdout_months": h, "train_end": str(cutoff.date()), "test_start": str((cutoff + pd.DateOffset(months=1)).date()),
                 "test_end": str(last.date()), "n_train_rows": int(len(train)), "n_test_rows": int(len(test)),
                 "evaluation": "one-month-ahead on held-out months (each prediction uses actual history up to the prior month)",
                 "metrics": {}}
    if test.empty:
        out["skipped"] = "No held-out rows with three months of prior history."
        return out
    out["metrics"]["naive_last_month"] = score(test["y"], test["lag_1"])
    if "lag_12" in test:
        seasonal = test[test["lag_12"].notna()]
        out["metrics"]["seasonal_naive"] = score(seasonal["y"], seasonal["lag_12"])
    if use_xgboost:
        if len(train) < MIN_TRAIN_ROWS:
            out["xgboost_skipped"] = f"Only {len(train)} training rows before the holdout (need {MIN_TRAIN_ROWS})."
        else:
            pipe = make_pipeline(spec, seed).fit(train[spec.columns], train["y"])
            pred = pd.Series(np.clip(pipe.predict(test[spec.columns]), 0, None), index=test.index)
            out["metrics"]["xgboost"] = score(test["y"], pred)
            if "seasonal_naive" in out["metrics"]:
                mask = test["lag_12"].notna()
                out["metrics"]["xgboost_on_seasonal_rows"] = score(test.loc[mask, "y"], pred[mask])
    return out


def fit_final(panel: pd.DataFrame, spec: FeatureSpec, seed: int) -> tuple[Pipeline, pd.DataFrame]:
    rows = usable(build_features(panel, spec))
    return make_pipeline(spec, seed).fit(rows[spec.columns], rows["y"]), rows


def live_series(panel: pd.DataFrame) -> pd.DataFrame:
    """Series observed in the dataset's final month; older series are not forecast from stale history."""
    last = panel["month_start"].max()
    return panel[(panel["month_start"] == last) & panel["units_sold"].notna()][KEY].drop_duplicates()


def recursive_forecast(panel: pd.DataFrame, spec: FeatureSpec, pipe: Pipeline, horizon: int = HORIZON) -> tuple[pd.DataFrame, int]:
    """Returns (forecast rows, number of live series skipped for missing recent lags)."""
    last = panel["month_start"].max()
    keys = live_series(panel)
    history = panel.merge(keys, on=KEY)
    held = history[history["month_start"] == last].set_index(KEY)[spec.drivers]
    outputs, skipped = [], 0
    for step in range(1, horizon + 1):
        target = last + pd.DateOffset(months=step)
        stub = held.reset_index().assign(month_start=target, units_sold=np.nan)
        feats = build_features(pd.concat([history, stub], ignore_index=True), spec)
        rows = feats[feats["target_month"] == target]
        ok = rows[REQUIRED_LAGS].notna().all(axis=1)
        if step == 1:
            skipped = int((~ok).sum())
        rows = rows[ok]
        if rows.empty:
            break
        preds = np.clip(pipe.predict(rows[spec.columns]), 0, None)
        step_out = rows[KEY].assign(target_month=target, step=step, y_pred=preds)
        outputs.append(step_out)
        filled = step_out[KEY].assign(month_start=target, units_sold=preds).merge(held.reset_index(), on=KEY, how="left")
        history = pd.concat([history, filled], ignore_index=True)
    out = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame(columns=KEY + ["target_month", "step", "y_pred"])
    return out, skipped


def naive_forecast(panel: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Last observed month repeated: the reference every model has to beat."""
    last = panel["month_start"].max()
    base = panel[(panel["month_start"] == last) & panel["units_sold"].notna()][KEY + ["units_sold"]]
    return pd.concat([base[KEY].assign(target_month=last + pd.DateOffset(months=s), step=s, y_pred=base["units_sold"].to_numpy())
                      for s in range(1, horizon + 1)], ignore_index=True)


def shap_importance(pipe: Pipeline, rows: pd.DataFrame, spec: FeatureSpec, seed: int) -> list[dict]:
    """Global mean |SHAP| per input (one-hot region/product columns summed back), via XGBoost's TreeSHAP."""
    sample = rows.sample(n=min(SHAP_SAMPLE, len(rows)), random_state=seed)
    pre = pipe.named_steps["pre"]
    X = np.asarray(pre.transform(sample[spec.columns]), dtype=float)
    names = list(pre.get_feature_names_out())
    onehot = set(pre.named_transformers_["ids"].get_feature_names_out())
    contrib = pipe.named_steps["model"].get_booster().predict(DMatrix(X), pred_contribs=True)[:, :-1]
    groups = ["region" if n in onehot and n.startswith("region_") else "product" if n in onehot else n for n in names]
    per_group = pd.DataFrame(contrib, columns=names).T.groupby(groups).sum().T  # samples x inputs
    abs_mean, mean = per_group.abs().mean(), per_group.mean()
    out = [{"feature": f, "mean_abs_shap": round(float(abs_mean[f]), 4), "mean_shap": round(float(mean[f]), 4)}
           for f in per_group.columns]
    return sorted(out, key=lambda r: -r["mean_abs_shap"])
