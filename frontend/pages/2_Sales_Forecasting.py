"""1-3 month unit forecasts per region with 80% intervals and 2025 backtest accuracy."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services import api_client
from services.ui import PRODUCTS, REGIONS, call, page

page("Sales Forecasting")

with st.sidebar:
    st.subheader("Forecast settings")
    region = st.selectbox("Region", list(REGIONS), format_func=lambda r: f"{REGIONS[r]} ({r})")
    model = st.radio("Model", ["xgboost", "prophet"], format_func={"xgboost": "XGBoost (live)", "prophet": "Prophet (precomputed)"}.get)
    horizon = st.slider("Horizon (months)", 1, 3, 3)

fc = call(api_client.forecast, region, horizon, model, spinner="Computing forecast...")
st.caption(f"Model `{fc['model_version']}` · {fc['generated']} · last observed month {fc['origin_month']} · "
           f"units = packs sold")

points, hist = pd.DataFrame(fc["forecast"]), pd.DataFrame(fc["history"])
if points.empty:
    st.info("The API returned no forecast rows for this selection.")
    st.stop()

for pid in sorted(points["product_id"].unique()):
    p, h = points[points["product_id"] == pid], hist[hist["product_id"] == pid]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["month"], y=h["units_sold"], name="Actual", mode="lines+markers", line={"color": "#1f77b4"}))
    if p["lower_units"].notna().all():
        fig.add_trace(go.Scatter(x=list(p["target_month"]) + list(p["target_month"])[::-1],
                                 y=list(p["upper_units"]) + list(p["lower_units"])[::-1], fill="toself", mode="lines",
                                 fillcolor="rgba(255,127,14,0.18)", line={"width": 0}, name="80% interval", hoverinfo="skip"))
    bridge = h.tail(1).rename(columns={"month": "target_month", "units_sold": "predicted_units"})
    line = pd.concat([bridge[["target_month", "predicted_units"]], p[["target_month", "predicted_units"]]])
    fig.add_trace(go.Scatter(x=line["target_month"], y=line["predicted_units"], name="Forecast",
                             mode="lines+markers", line={"color": "#ff7f0e", "dash": "dash"}))
    fig.update_layout(title=f"{REGIONS[region]} · {PRODUCTS[pid]}", height=320, margin={"t": 50, "b": 20},
                      yaxis_title="Units", legend={"orientation": "h", "y": -0.2})
    st.plotly_chart(fig, width="stretch")

st.subheader("Forecast table")
st.dataframe(points.drop(columns=["product_id"]).rename(columns={
    "product_name": "Product", "target_month": "Month", "step": "Months ahead", "predicted_units": "Forecast",
    "lower_units": "Lower (80%)", "upper_units": "Upper (80%)"}), hide_index=True, width="stretch")

st.subheader("How accurate was this model? (2025 backtest, all regions)")
acc = pd.DataFrame(fc["backtest_accuracy_2025"])
if not acc.empty:
    st.dataframe(acc.rename(columns={"horizon_months": "Months ahead", "mae_units": "MAE (units)", "rmse_units": "RMSE (units)",
                                     "wape_pct": "WAPE %", "n_predictions": "Predictions"}), hide_index=True, width="stretch")
cov = fc["interval"]["empirical_coverage_2025"]
level = fc["interval"]["level"]
if cov:
    worst = min(cov.values())
    msg = (f"Nominal {level:.0%} interval; share of 2025 actuals that fell inside it: "
           + ", ".join(f"{h} month ahead {v:.0%}" for h, v in sorted(cov.items(), key=lambda kv: int(kv[0]))) + ".")
    (st.warning if worst < level - 0.1 else st.info)(
        msg + (" The interval is too narrow: treat it as optimistic." if worst < level - 0.1 else ""))
st.caption("Method: " + fc["interval"]["method"])

with st.expander("Assumptions behind this forecast", expanded=True):
    for a in fc["assumptions"]:
        st.markdown(f"- {a}")
