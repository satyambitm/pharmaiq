"""Scenario-based budget allocation under ASSUMED concave response curves."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from components import banner
from src.optimization import allocate
from src.optimization.response import DEFAULT_RESPONSES, SCENARIO_ALPHA_MULTIPLIER, apply_scenario

st.set_page_config(page_title="Budget Simulator", layout="wide")
st.title("Marketing budget simulator")
banner()
st.error("Model outputs are illustrative and not observed causal uplift or guaranteed return. Response parameters "
         "are hand-set assumptions, not estimated from data.")

DEFAULTS = {"budget": 100.0, "scenario": "base", "min_cr": 5.0, "max_cr": 30.0, "pressure_regions": [], "pressure": 0.3}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
if st.sidebar.button("Reset defaults"):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.rerun()

st.sidebar.number_input("Total budget (INR crore)", min_value=0.0, max_value=500.0, step=5.0, key="budget")
st.sidebar.selectbox("Response scenario", list(SCENARIO_ALPHA_MULTIPLIER), key="scenario")
st.sidebar.number_input("Regional minimum (INR crore)", min_value=0.0, max_value=100.0, step=1.0, key="min_cr")
st.sidebar.number_input("Regional maximum (INR crore)", min_value=0.0, max_value=200.0, step=1.0, key="max_cr")
st.sidebar.multiselect("Regions under competitor pressure", [r.region for r in DEFAULT_RESPONSES], key="pressure_regions")
st.sidebar.slider("Competitor pressure (alpha reduction)", 0.0, 0.9, step=0.05, key="pressure")

s = st.session_state
responses = tuple(replace(r, min_spend=s.min_cr, max_spend=s.max_cr) for r in DEFAULT_RESPONSES)
responses = apply_scenario(responses, s.scenario, {r: s.pressure for r in s.pressure_regions})

with st.expander("Assumed response equation and parameters", expanded=True):
    st.latex(r"\text{gain}_r(b_r) = \alpha_r \ln\!\left(1 + \frac{b_r}{\text{scale}_r}\right)\quad\text{(INR crore)}")
    st.dataframe(pd.DataFrame([r.__dict__ for r in responses]).round(3), hide_index=True, use_container_width=True)
    st.caption(f"Scenario multiplier on alpha: {SCENARIO_ALPHA_MULTIPLIER[s.scenario]}. A uniform multiplier changes total "
               "gain but not the optimal split; regional competitor pressure does change the split.")

lo, hi = s.min_cr * len(responses), s.max_cr * len(responses)
try:
    table = allocate.compare_strategies(s.budget, responses)
except allocate.InfeasibleBudgetError as exc:
    st.error(f"Infeasible: {exc}. Budget must be between {lo:.0f} and {hi:.0f} crore with these bounds.")
    st.stop()

tot = table.groupby("strategy").agg(spend_cr=("spend_cr", "sum"), assumed_gain_cr=("assumed_gain_cr", "sum"),
                                    all_within_bounds=("within_bounds", "all")).reset_index()
opt, eq = tot.set_index("strategy").loc["optimized"], tot.set_index("strategy").loc["equal_split"]
c1, c2, c3 = st.columns(3)
c1.metric("Assumed incremental revenue (optimized)", f"₹{opt['assumed_gain_cr']:,.2f} cr")
c2.metric("vs equal split", f"₹{opt['assumed_gain_cr'] - eq['assumed_gain_cr']:+,.2f} cr")
c3.metric("Assumed incremental revenue per ₹1 spent", f"₹{opt['assumed_gain_cr'] / s.budget:,.3f}" if s.budget else "—")

st.plotly_chart(px.bar(table, x="region", y="spend_cr", color="strategy", barmode="group",
                       title="Allocation by region (INR crore)", labels={"spend_cr": "INR crore"}), use_container_width=True)
st.dataframe(tot.round(3), hide_index=True, use_container_width=True)
if not tot["all_within_bounds"].all():
    st.warning("Some comparison strategies violate the regional bounds (see `within_bounds` below).")
st.dataframe(table.round(3), hide_index=True, use_container_width=True)
st.caption("At the optimum, regions not at a bound share the same marginal gain per crore — that is the optimality condition.")

st.subheader("Sensitivity: total budget")
budgets = list(np.linspace(lo, hi, 15))
sens = allocate.budget_sensitivity(budgets, responses)
st.plotly_chart(px.line(sens.melt(id_vars="budget_cr", value_vars=["optimized_gain_cr", "equal_split_gain_cr"]),
                        x="budget_cr", y="value", color="variable", markers=True,
                        labels={"budget_cr": "Budget (INR crore)", "value": "Assumed incremental revenue (INR crore)"}),
                use_container_width=True)
