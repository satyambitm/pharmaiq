"""Provenance, evaluation design, leakage safeguards, limitations, versions."""

from __future__ import annotations

import streamlit as st

from components import ROOT, banner, load_json
from src import config
from src.features.build import FEATURES

st.set_page_config(page_title="Methodology", layout="wide")
st.title("Methodology")
banner()

meta_path = config.METADATA_JSON
if meta_path.exists():
    meta = load_json(meta_path)
    st.subheader("Data provenance")
    st.markdown(f"- Version `{meta['dataset_version']}`, equation `{meta['equation_version']}`, seed `{meta['seed']}`\n"
                f"- Rows: {meta['n_rows_clean']:,} clean; dirty practice copy {meta['n_rows_dirty']:,} rows with "
                f"{len(meta['anomalies'])} logged anomalies (never used for modeling)\n"
                f"- Clean CSV SHA-256: `{meta['clean_sha256']}`\n- Price convention: {meta['price_convention']}\n"
                f"- Competitor index: {meta['competitor_index_scale']}")

st.subheader("Forecast evaluation")
st.markdown(f"""
- Target: units sold in month T for each region × product; features use data observed by end of month T−1.
- Tuning: 3 expanding folds (validation Jul–Dec 2023, Jan–Jun 2024, Jul–Dec 2024). 2025 never used for tuning.
- Final: refit on targets through Dec 2024; score once on Jan–Dec 2025 (216 rows) for every model.
- Models: seasonal naive (same month last year), last-month naive, XGBoost (one-hot ids + history features),
  Prophet (univariate per series, refit at every origin on history only, `ds`/`y`).
- Metrics: MAE, RMSE, WAPE in units; overall and per region.
- XGBoost features: `{", ".join(FEATURES)}`
""")
st.subheader("Leakage safeguards")
st.markdown("""
- Every lag and rolling statistic is computed with `groupby().shift(1)` before `rolling()`.
- Target-month spend, price, revenue and competitor index are never inputs.
- Unit test perturbs future months massively and asserts earlier features are unchanged.
- K-Means descriptors use only months ≤ Dec 2023; scaler fit on those descriptors only.
""")
st.subheader("Limitations")
st.markdown("""
- Data is synthetic with a known generating equation; good accuracy is expected and is not evidence about real markets.
- 18 market cells is too few for robust clustering; segments are exploratory.
- SHAP shows model associations; correlated lags can split importance; nothing here is causal.
- Budget response curves are assumptions; a real plan would need randomized or quasi-experimental evidence.
""")
run = config.METRICS_DIR / "run_metadata.json"
if run.exists():
    st.subheader("Package versions (last training run)")
    st.json(load_json(run)["packages"])
st.markdown("Docs in the repo: `docs/model_card.md`, `docs/assumptions.md`, `docs/data_dictionary.md`, `sql/analysis_queries.sql`.")
