"""Chronological backtest scorecards, interval coverage and SHAP explanations."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from services import api_client
from services.ui import PRODUCTS, call, page

page("Model Insights")

ins = call(api_client.model_insights)
st.caption(ins["evaluation"])

sc = pd.DataFrame(ins["horizon_scorecard"])
st.subheader("Forecast accuracy by horizon (2025 test period)")
st.plotly_chart(px.bar(sc, x="horizon_months", y="wape_pct", color="model", barmode="group",
                       labels={"horizon_months": "Months ahead", "wape_pct": "WAPE % (lower is better)"},
                       title="WAPE by model and horizon"), width="stretch")
st.dataframe(sc.rename(columns={"model": "Model", "horizon_months": "Months ahead", "n_predictions": "Predictions",
                                "mae_units": "MAE (units)", "rmse_units": "RMSE (units)", "wape_pct": "WAPE %"}),
             hide_index=True, width="stretch")
st.caption("WAPE = total absolute error / total actual units. It is volume-weighted, so large series dominate it.")

if ins["one_step_scorecard"]:
    with st.expander("One-month-ahead backtest folds (model selection, 2023-2024) and 2025 test"):
        one = pd.DataFrame(ins["one_step_scorecard"])
        cols = [c for c in ["model", "evaluation", "test_start", "test_end", "n_predictions", "MAE_units", "RMSE_units", "WAPE_%"] if c in one]
        st.dataframe(one[cols].round(2), hide_index=True, width="stretch")

st.subheader("Interval calibration")
cov = pd.DataFrame(ins["interval_coverage"])
if not cov.empty:
    st.dataframe(cov.rename(columns={"model": "Model", "horizon_months": "Months ahead", "nominal": "Nominal",
                                     "empirical_coverage": "Actual coverage 2025", "n": "Predictions",
                                     "mean_width_units": "Mean width (units)"}), hide_index=True, width="stretch")
    st.caption("A well-calibrated 80% interval should contain about 80% of actuals. Lower coverage means the band is too narrow.")

st.subheader("What drives XGBoost's predictions (SHAP)")
st.info(ins["shap_caveat"], icon="ℹ️")
imp = pd.DataFrame(ins["global_importance"])
st.plotly_chart(px.bar(imp.sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature", orientation="h",
                       labels={"mean_abs_shap": "Mean |SHAP| (units)", "feature": ""},
                       title="Global importance on 2025 one-step holdout rows"), width="stretch")

examples = ins["local_examples"]
if examples:
    labels = [f"{e['region_id']} · {PRODUCTS.get(e['product_id'], e['product_id'])} · {e['target_month']}" for e in examples]
    pick = st.selectbox("Explain one prediction", range(len(examples)), format_func=labels.__getitem__)
    e = examples[pick]
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline (average prediction)", f"{e['base_value']:,.0f}")
    c2.metric("Prediction", f"{e['prediction']:,.0f}")
    c3.metric("Actual", f"{e['actual_units']:,.0f}")
    contrib = pd.DataFrame(e["contributions"] + [{"feature": "all other features", "feature_value": float("nan"),
                                                  "shap": e["other_features_shap"]}])
    st.plotly_chart(px.bar(contrib, x="shap", y="feature", orientation="h", color=contrib["shap"] > 0,
                           color_discrete_map={True: "#2ca02c", False: "#d62728"},
                           labels={"shap": "Contribution to prediction (units)", "feature": "", "color": "Pushes up"},
                           title="Feature contributions for this prediction"), width="stretch")

with st.expander("Model versions"):
    st.json(ins["model_versions"])
