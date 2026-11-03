"""K-Means segments of region x product market cells."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from services import api_client
from services.ui import call, page

page("Market Segmentation")

seg = call(api_client.segments)
cells, profiles, ks = pd.DataFrame(seg["cells"]), pd.DataFrame(seg["profiles"]), pd.DataFrame(seg["k_selection"])
if cells.empty:
    st.info("The API returned no segments.")
    st.stop()

st.caption(f"K = {seg['k']} · {len(cells)} market cells · descriptors use history up to {seg['history_cutoff']} only")
cells["cell"] = cells["region_name"] + " · " + cells["product_name"]
left, right = st.columns([3, 2])
left.plotly_chart(px.scatter(cells, x="pc1", y="pc2", color="segment_label", symbol="product_name", hover_name="cell",
                             title="Market cells on the first two principal components",
                             labels={"pc1": "PC1", "pc2": "PC2", "segment_label": "Segment"}), width="stretch")
right.plotly_chart(px.line(ks, x="k", y="silhouette", markers=True, title="Silhouette score by K"), width="stretch")
right.dataframe(ks[["k", "silhouette", "min_cluster_size", "chosen"]].round(3), hide_index=True, width="stretch")

st.subheader("Segment profiles (averages per segment)")
st.dataframe(profiles.drop(columns=["cluster"]).round(2).rename(columns={
    "segment_label": "Segment", "n_cells": "Cells", "avg_monthly_units": "Avg monthly units", "cv_units": "Volatility (CV)",
    "yoy_growth_pct": "YoY growth %", "avg_discount_pct": "Discount %", "spend_per_unit_inr": "Spend per unit (INR)",
    "seasonality_amplitude": "Seasonality", "campaign_frequency": "Campaign share"}), hide_index=True, width="stretch")

st.subheader("Cell assignments")
st.dataframe(cells[["region_name", "product_name", "segment_label", "avg_monthly_units", "cv_units", "yoy_growth_pct"]]
             .sort_values(["segment_label", "region_name"]).round(2), hide_index=True, width="stretch")

with st.expander("Method and caveats", expanded=True):
    st.markdown(f"Features (standardised): {', '.join(f'`{f}`' for f in seg['features'])}")
    for n in seg["notes"]:
        st.markdown(f"- {n}")
