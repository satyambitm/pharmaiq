"""PharmaIQ Lite+ dashboard — overview. Run: streamlit run dashboard/app.py"""

from __future__ import annotations

import streamlit as st

from components import CR, banner, load_json, load_panel
from src import config

st.set_page_config(page_title="PharmaIQ Lite+", page_icon="💊", layout="wide")
st.title("PharmaIQ Lite+ — Aster Pharma commercial analytics")
banner()

df = load_panel()
if df is None:
    st.stop()

df["year"] = df["month_start"].dt.year
yearly = df.groupby("year")[["net_revenue_inr", "units_sold"]].sum()
last, prev = yearly.index.max(), yearly.index.max() - 1

c1, c2, c3, c4 = st.columns(4)
c1.metric("Net revenue, all years", f"₹{df['net_revenue_inr'].sum() / CR:,.1f} cr")
c2.metric("Units sold, all years", f"{df['units_sold'].sum():,.0f}")
c3.metric(f"Net revenue {last}", f"₹{yearly.loc[last, 'net_revenue_inr'] / CR:,.1f} cr",
          f"{(yearly.loc[last, 'net_revenue_inr'] / yearly.loc[prev, 'net_revenue_inr'] - 1) * 100:+.1f}% YoY")
c4.metric(f"Units {last}", f"{yearly.loc[last, 'units_sold']:,.0f}",
          f"{(yearly.loc[last, 'units_sold'] / yearly.loc[prev, 'units_sold'] - 1) * 100:+.1f}% YoY")

st.subheader("Dataset")
st.markdown(
    f"""
- **Grain:** one row per month × region × product — {df['region_id'].nunique()} regions × {df['product_id'].nunique()} products × {df['month_start'].nunique()} months = **{len(df):,} rows**
- **Coverage:** {df['month_start'].min():%b %Y} – {df['month_start'].max():%b %Y}
- **Source:** deterministic generator `src/data/generate.py`, version `{config.DATA_SOURCE}`, equation `{config.EQUATION_VERSION}`
- **Forecast evaluation:** train ≤ Dec 2023, validation 2024, untouched test Jan–Dec 2025 (one month ahead)
"""
)

meta_path = config.METRICS_DIR / "run_metadata.json"
if meta_path.exists():
    meta = load_json(meta_path)
    st.subheader("Last training run")
    st.markdown(f"- Run at (UTC): `{meta['run_at_utc']}`  \n- XGBoost params: `{meta['xgboost_best_params']}`  \n"
                f"- Prophet included: `{meta['prophet_run']}`  \n- Python {meta['packages']['python']}, "
                f"xgboost {meta['packages']['xgboost']}, prophet {meta['packages']['prophet']}")
else:
    st.info("No training run found yet. Run `python scripts/train_all.py` to populate the Forecasting page.")

st.subheader("Pages")
st.markdown(
    """
1. **Performance** — SQL-backed trends, regional shares, product ranks, discounts
2. **Forecasting** — seasonal naive vs XGBoost vs Prophet on identical 2025 rows, SHAP
3. **Segmentation** — descriptive K-Means market cells (training period only)
4. **Budget Simulator** — constrained allocation under *assumed* response curves
5. **Methodology** — provenance, leakage safeguards, limitations
"""
)
st.subheader("Limitations")
st.markdown("Synthetic relationships are known by construction, so models recovering them is expected and says nothing "
            "about real markets. Segments use 18 cells and are exploratory. The budget simulator uses hand-set "
            "response parameters, not estimated causal effects.")
