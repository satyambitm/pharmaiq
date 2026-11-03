"""Descriptive K-Means segments of region x product market cells (history through Dec 2023)."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from components import banner, load_csv, missing
from src import config
from src.segmentation.kmeans import SEGMENT_FEATURES

st.set_page_config(page_title="Segmentation", layout="wide")
st.title("Market-cell segmentation (K-Means)")
banner()

seg_path = config.CLUSTERS_DIR / "market_segments.csv"
if missing(seg_path, "python scripts/build_reports.py"):
    st.stop()
seg = load_csv(seg_path)
prof = load_csv(config.CLUSTERS_DIR / "cluster_profiles.csv")
cz = load_csv(config.CLUSTERS_DIR / "cluster_centers_standardized.csv")
kt = load_csv(config.CLUSTERS_DIR / "k_selection.csv")

st.info(f"18 market cells · descriptors from Jan 2021 – {config.SEGMENTATION_CUTOFF:%b %Y} only · "
        "segments describe past behavior; they are not causal drivers and say nothing about ROI. "
        "With N = 18, treat them as exploratory.")

c1, c2 = st.columns([2, 1])
seg["cell"] = seg["region_name"] + " / " + seg["product_name"]
c1.plotly_chart(px.scatter(seg, x="pc1", y="pc2", color="segment_label", hover_name="cell", text="cell",
                           title="PCA projection (visual aid only, not proof of separation)"), use_container_width=True)
c2.markdown("**K selection**")
c2.dataframe(kt.round(3), hide_index=True, use_container_width=True)

st.subheader("Standardized centroids (z-scores)")
st.plotly_chart(px.imshow(cz.set_index("segment_label")[SEGMENT_FEATURES], color_continuous_scale="RdBu_r",
                          zmin=-2, zmax=2, aspect="auto"), use_container_width=True)

st.subheader("Segment profiles (original units)")
st.dataframe(prof.round(2), hide_index=True, use_container_width=True)
st.subheader("Cell assignments")
st.dataframe(seg[["region_name", "product_name", "segment_label"] + SEGMENT_FEATURES].round(2), hide_index=True, use_container_width=True)
with st.expander("Feature definitions"):
    st.markdown("""
- **avg_monthly_units**: mean units per month
- **cv_units**: std / mean of monthly units (volatility, includes trend and seasonality)
- **yoy_growth_pct**: 2023 vs 2022 total units
- **avg_discount_pct**: total discount / total gross revenue
- **spend_per_unit_inr**: total marketing spend / total units (descriptive ratio, not efficiency)
- **seasonality_amplitude**: (max − min) / mean of the calendar-month average profile
- **campaign_frequency**: share of months flagged as campaign
""")
