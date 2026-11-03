"""Scenario-based regional budget allocation under ASSUMED response curves."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from services import api_client
from services.ui import REGIONS, call, page

page("Budget Optimizer")
st.error("Scenario simulation only. Response curves are hand-set assumptions, not estimates from data, so results "
         "are not measured ROI, causal uplift or a guaranteed return.")

with st.sidebar:
    st.subheader("Scenario")
    budget = st.number_input("Total budget (INR crore)", min_value=1.0, max_value=1000.0, value=100.0, step=5.0)
    scenario = st.selectbox("Response scenario", ["base", "low", "high"])
    lo = st.number_input("Minimum per region (INR crore)", min_value=0.0, max_value=200.0, value=5.0, step=1.0)
    hi = st.number_input("Maximum per region (INR crore)", min_value=1.0, max_value=500.0, value=30.0, step=1.0)
    pressured = st.multiselect("Regions under competitor pressure", list(REGIONS.values()))
    pressure = st.slider("Competitor pressure (response reduction)", 0.0, 0.9, 0.3, 0.05)

feasible = (lo * len(REGIONS), hi * len(REGIONS))
if lo > hi:
    st.error("The regional minimum cannot exceed the maximum.")
    st.stop()
if not feasible[0] <= budget <= feasible[1]:
    st.error(f"With these bounds the budget must be between ₹{feasible[0]:,.0f} and ₹{feasible[1]:,.0f} crore.")
    st.stop()

res = call(api_client.optimize_budget, budget, scenario, tuple((r, pressure) for r in pressured), lo, hi,
           spinner="Optimising allocation...")
st.caption(res["label"])

strategies = {s["strategy"]: s for s in res["strategies"]}
opt = strategies["optimized"]
c1, c2, c3 = st.columns(3)
c1.metric("Assumed incremental revenue (optimised)", f"₹{opt['total_assumed_gain_cr']:,.2f} cr")
c2.metric("vs equal split", f"₹{res['optimized_vs_equal_split_cr']:+,.2f} cr")
c3.metric("vs status quo", f"₹{res['optimized_vs_status_quo_cr']:+,.2f} cr")

alloc = pd.DataFrame([{**a, "strategy": s["strategy"]} for s in res["strategies"] for a in s["allocation"]])
st.plotly_chart(px.bar(alloc, x="region", y="spend_cr", color="strategy", barmode="group",
                       labels={"spend_cr": "INR crore", "region": ""}, title="Allocation by region"), width="stretch")
broken = [s["strategy"] for s in res["strategies"] if not s["feasible"]]
if broken:
    st.warning(f"These comparison strategies break the regional bounds: {', '.join(broken)}.")
st.dataframe(pd.DataFrame(opt["allocation"]).round(3).rename(columns={
    "region": "Region", "spend_cr": "Spend (cr)", "assumed_gain_cr": "Assumed gain (cr)",
    "marginal_gain_per_cr": "Marginal gain per cr", "min_cr": "Min", "max_cr": "Max", "within_bounds": "Within bounds"}),
    hide_index=True, width="stretch")
st.caption("At the optimum, regions not at a bound have equal marginal gain per crore. That is the optimality condition.")

st.subheader("Sensitivity to total budget")
sens = pd.DataFrame(res["sensitivity"])
if not sens.empty:
    st.plotly_chart(px.line(sens.melt(id_vars="budget_cr", value_vars=["optimized_gain_cr", "equal_split_gain_cr"]),
                            x="budget_cr", y="value", color="variable", markers=True,
                            labels={"budget_cr": "Budget (INR crore)", "value": "Assumed incremental revenue (INR crore)",
                                    "variable": ""}), width="stretch")

with st.expander("Objective, constraints and assumptions", expanded=True):
    st.latex(r"\max_{b}\ \sum_r \alpha_r \ln\!\left(1 + \frac{b_r}{s_r}\right)\quad \text{s.t.}\ \sum_r b_r = B,\ \ "
             r"\text{min} \le b_r \le \text{max}")
    st.dataframe(pd.DataFrame(res["response_curves"]), hide_index=True, width="stretch")
    for a in res["assumptions"]:
        st.markdown(f"- {a}")
