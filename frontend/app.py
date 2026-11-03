"""PharmaIQ Streamlit frontend (talks only to the FastAPI backend).

Local:  streamlit run frontend/app.py         (with the API on http://localhost:8000)
Cloud:  set API_BASE_URL to the public HTTPS backend URL in the app's Secrets.
"""

from __future__ import annotations

import streamlit as st

from services import api_client
from services.ui import page

page("PharmaIQ: pharma sales analytics")

ready = api_client.readiness()
c1, c2, c3 = st.columns(3)
c1.metric("Backend", ready.get("status", "unknown"))
c2.metric("Data rows", f"{ready['data_rows']:,}" if ready.get("data_rows") else "n/a")
c3.metric("Data window", ready.get("data_window") or "n/a")
if ready.get("model_versions"):
    st.caption("Model versions: " + ", ".join(f"{k} `{v}`" for k, v in ready["model_versions"].items()))
if ready.get("status") != "ready":
    st.info("If the API runs on a free hosting tier it sleeps after about 15 minutes idle. The first request wakes "
            "it, which can take about a minute. Use **Retry connection** in the sidebar.")

st.subheader("Pages")
st.markdown("""
1. **Overview**: portfolio KPIs, monthly trend, product mix and regional performance
2. **Sales Forecasting**: 1-3 month forecasts per region from XGBoost (live) or Prophet (precomputed), with 80% intervals
3. **Market Segmentation**: K-Means segments of the 18 region × product market cells
4. **Model Insights**: chronological backtest scores, interval coverage and SHAP explanations
5. **Budget Optimizer**: scenario-based regional allocation under *assumed* response curves
""")

st.subheader("How it works")
st.markdown("""
- This Streamlit app calls a **FastAPI** backend over HTTPS; it never reads the database or model files itself.
- The backend reads monthly sales from **PostgreSQL** (or a bundled CSV fallback) and serves models that were
  trained offline and committed as a versioned bundle. Nothing is retrained per request.
- Grain: month × region × product, Jan 2021 to Dec 2025 (6 regions × 3 products × 60 months = 1,080 rows).
""")
st.caption("Limitations: the data is synthetic, so models recovering its built-in patterns says nothing about real "
           "markets. SHAP shows associations, not causes. Budget results depend entirely on hand-set response curves.")
