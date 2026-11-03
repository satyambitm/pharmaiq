"""Historical performance: trends, regional share, product ranks, discounts, SQL drilldown."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from components import CR, banner, caption_scope, filters, load_panel
from src.analytics import sql_reporting
from src.data.database import db_available

st.set_page_config(page_title="Performance", layout="wide")
st.title("Performance")
banner()

df = load_panel()
if df is None:
    st.stop()
f = filters(df, "perf")
caption_scope(f)
if f.empty:
    st.stop()

monthly = f.groupby("month_start")[["units_sold", "net_revenue_inr", "discount_inr", "gross_revenue_inr"]].sum().reset_index()
monthly["net_revenue_cr"] = monthly["net_revenue_inr"] / CR
monthly["discount_pct"] = monthly["discount_inr"] / monthly["gross_revenue_inr"] * 100

c1, c2 = st.columns(2)
c1.plotly_chart(px.line(monthly, x="month_start", y="units_sold", title="Monthly units sold",
                        labels={"month_start": "Month", "units_sold": "Units"}), use_container_width=True)
c2.plotly_chart(px.line(monthly, x="month_start", y="net_revenue_cr", title="Monthly net revenue (INR crore)",
                        labels={"month_start": "Month", "net_revenue_cr": "INR crore"}), use_container_width=True)

reg = f.groupby("region_name")["net_revenue_inr"].sum().reset_index()
reg["share_pct"] = reg["net_revenue_inr"] / reg["net_revenue_inr"].sum() * 100
prod = f.groupby(["region_name", "product_name"])["net_revenue_inr"].sum().reset_index()
prod["net_revenue_cr"] = prod["net_revenue_inr"] / CR
c3, c4 = st.columns(2)
c3.plotly_chart(px.pie(reg, names="region_name", values="share_pct", title="Regional share of net revenue (%)"), use_container_width=True)
c4.plotly_chart(px.bar(prod, x="region_name", y="net_revenue_cr", color="product_name", barmode="group",
                       title="Product net revenue by region (INR crore)",
                       labels={"region_name": "Region", "net_revenue_cr": "INR crore", "product_name": "Product"}),
                use_container_width=True)

st.plotly_chart(px.line(monthly, x="month_start", y="discount_pct", title="Weighted discount rate (% of gross revenue)",
                        labels={"month_start": "Month", "discount_pct": "Discount %"}), use_container_width=True)

st.subheader("SQL drilldown (PostgreSQL)")
if db_available():
    queries = sql_reporting.load_queries()
    name = st.selectbox("Query", list(queries), format_func=lambda n: f"{n} — {queries[n]['question'][:80]}")
    st.caption(queries[name]["question"])
    with st.expander("SQL"):
        st.code(queries[name]["sql"], language="sql")
    st.dataframe(sql_reporting.run_query(name), use_container_width=True)
    st.caption("Runs on the full warehouse table; sidebar filters do not apply to SQL results.")
else:
    st.info("PostgreSQL not reachable. Start it with `docker compose up -d` and run `python scripts/bootstrap.py`.")
