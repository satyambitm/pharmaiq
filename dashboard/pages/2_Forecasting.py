"""Forecast comparison on identical 2025 holdout rows, plus SHAP and an illustrative next-month forecast."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from components import banner, load_csv, load_parquet, missing
from src import config

st.set_page_config(page_title="Forecasting", layout="wide")
st.title("Forecasting — one month ahead, units")
banner()

pred_path = config.FORECASTS_DIR / "holdout_predictions.parquet"
card_path = config.METRICS_DIR / "model_comparison.csv"
if missing(pred_path, "python scripts/train_all.py") or missing(card_path, "python scripts/train_all.py"):
    st.stop()

preds = load_parquet(pred_path).copy()
preds["region_name"] = preds["region_id"].map(config.REGIONS)
preds["product_name"] = preds["product_id"].map(config.PRODUCTS)
card = load_csv(card_path)

st.subheader("Scorecard (all models scored on exactly the same rows)")
evaluation = st.radio("Evaluation window", sorted(card["evaluation"].unique(), key=lambda e: e != "test_2025"), horizontal=True)
st.dataframe(card[card["evaluation"] == evaluation].drop(columns=["evaluation"]).round(3), use_container_width=True, hide_index=True)
st.caption("Horizon: 1 month ahead. Scope: all 18 region × product series. Lower is better. XGBoost hyperparameters were "
           "chosen on the backtest folds, so its fold scores are optimistic; test_2025 was never used for tuning.")

st.subheader("Holdout 2025: actual vs predicted")
scope = st.selectbox("Scope", ["All series (sum)"] + sorted(f"{r} / {p}" for r, p in preds[["region_name", "product_name"]].drop_duplicates().itertuples(index=False)))
if scope == "All series (sum)":
    view = preds.groupby(["model", "target_month"])[["y", "y_pred"]].sum().reset_index()
else:
    r, p = scope.split(" / ")
    view = preds[(preds["region_name"] == r) & (preds["product_name"] == p)]
actual = view[view["model"] == view["model"].iloc[0]][["target_month", "y"]].rename(columns={"y": "units"}).assign(series="actual")
lines = pd.concat([actual, view[["target_month", "y_pred", "model"]].rename(columns={"y_pred": "units", "model": "series"})])
st.plotly_chart(px.line(lines, x="target_month", y="units", color="series", markers=True,
                        labels={"target_month": "Target month (forecast origin = previous month)", "units": "Units"}),
                use_container_width=True)

reg_path = config.METRICS_DIR / "metrics_by_region.csv"
if reg_path.exists():
    reg = load_csv(reg_path)
    reg = reg[reg["evaluation"] == "test_2025"].copy()
    reg["region_name"] = reg["region_id"].map(config.REGIONS)
    st.plotly_chart(px.bar(reg, x="region_name", y="WAPE_%", color="model", barmode="group",
                           title="2025 WAPE by region (%)", labels={"region_name": "Region"}), use_container_width=True)

st.subheader("SHAP — what the XGBoost model relied on (association, not causation)")
c1, c2 = st.columns(2)
for col, name in ((c1, "shap_summary.png"), (c2, "shap_waterfall_example.png")):
    p = config.FIGURES_DIR / name
    if p.exists():
        col.image(str(p))
st.caption("SHAP shows how features moved this model's predictions on 2025 rows. It cannot tell you that changing "
           "marketing spend would cause a sales change.")

nxt = config.FORECASTS_DIR / "next_month_forecast.csv"
if nxt.exists():
    st.subheader("Illustrative next-month forecast (Jan 2026)")
    st.error("Future forecast — no actuals exist. Revenue uses an ASSUMED price and discount, not observed values.")
    f = load_csv(nxt)
    f["region_name"] = f["region_id"].map(config.REGIONS)
    f["product_name"] = f["product_id"].map(config.PRODUCTS)
    st.dataframe(f[["target_month", "region_name", "product_name", "y_pred", "assumed_unit_price_inr",
                    "assumed_discount_rate", "estimated_net_revenue_inr"]].round(2), use_container_width=True, hide_index=True)
    st.caption(f["note"].iloc[0])
