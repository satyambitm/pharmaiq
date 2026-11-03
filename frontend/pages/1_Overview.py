"""Portfolio KPIs, monthly trend, product mix and regional performance."""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

from services import api_client
from services.ui import PRODUCTS, call, page

page("Overview")

MONTHS = [d.date() for d in pd.date_range("2021-01-01", "2025-12-01", freq="MS")]
with st.sidebar:
    st.subheader("Filters")
    start, end = st.select_slider("Month range", options=MONTHS, value=(MONTHS[0], MONTHS[-1]),
                                  format_func=lambda d: d.strftime("%b %Y"))
    products = st.multiselect("Products (regional view)", list(PRODUCTS), default=list(PRODUCTS),
                              format_func=PRODUCTS.get)


def iso(d: date) -> str:
    return d.isoformat()


ov = call(api_client.overview, iso(start), iso(end))
k = ov["kpis"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Net revenue", f"₹{k['net_revenue_cr']:,.1f} cr")
c2.metric("Units sold", f"{k['units_sold']:,}")
c3.metric("Marketing spend", f"₹{k['marketing_spend_cr']:,.1f} cr")
c4.metric("Avg discount", f"{k['avg_discount_pct']:.1f}%")
g = k["revenue_growth_pct_last12_vs_prior12"]
c5.metric("Revenue growth, last 12 vs prior 12 months", f"{g:+.1f}%" if g is not None else "n/a",
          help="Needs at least 24 months in the selected window")
st.caption(f"{ov['months']} months · {ov['window_start']} to {ov['window_end']} · money in INR crore (1 cr = 10 million)")

monthly = pd.DataFrame(ov["monthly"])
left, right = st.columns([2, 1])
left.plotly_chart(px.line(monthly, x="month", y=["net_revenue_cr", "marketing_spend_cr"], markers=False,
                          labels={"value": "INR crore", "month": "", "variable": ""},
                          title="Monthly net revenue and marketing spend"), width="stretch")
mix = pd.DataFrame(ov["product_mix"])
right.plotly_chart(px.pie(mix, names="product_name", values="net_revenue_cr", hole=0.45,
                          title="Net revenue mix by product"), width="stretch")

st.subheader("Regional performance")
if not products:
    st.info("Select at least one product to see regional performance.")
    st.stop()
rg = call(api_client.regions, iso(start), iso(end), tuple(products))
reg = pd.DataFrame(rg["regions"])
if reg.empty:
    st.info("No regional data for this selection.")
    st.stop()
left, right = st.columns(2)
left.plotly_chart(px.bar(reg, x="region_name", y="net_revenue_cr", title="Net revenue by region (INR crore)",
                         labels={"region_name": "", "net_revenue_cr": "INR crore"}), width="stretch")
right.plotly_chart(px.scatter(reg, x="marketing_spend_cr", y="net_revenue_cr", text="region_name", size="units_sold",
                              title="Spend vs revenue by region (association only, not ROI)",
                              labels={"marketing_spend_cr": "Marketing spend (INR crore)", "net_revenue_cr": "Net revenue (INR crore)"}),
                   width="stretch")
rm = pd.DataFrame(rg["monthly"]).merge(reg[["region_id", "region_name"]], on="region_id")
st.plotly_chart(px.line(rm, x="month", y="net_revenue_cr", color="region_name", title="Monthly net revenue by region",
                        labels={"net_revenue_cr": "INR crore", "month": "", "region_name": "Region"}), width="stretch")
st.dataframe(reg.drop(columns=["region_id"]).rename(columns={
    "region_name": "Region", "net_revenue_cr": "Net revenue (cr)", "units_sold": "Units", "marketing_spend_cr": "Spend (cr)",
    "spend_to_revenue_pct": "Spend / revenue %", "avg_discount_pct": "Discount %", "revenue_share_pct": "Revenue share %",
    "growth_pct_last12_vs_prior12": "Growth % (last 12 vs prior 12)"}), hide_index=True, width="stretch")
