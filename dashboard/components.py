"""Shared dashboard helpers: provenance banner, cached loaders, artifact guards, filters."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

CR = 1e7


def _mtime(path: Path) -> float:
    """Cache key component so cached tables refresh when an artifact file changes."""
    return path.stat().st_mtime if path.exists() else 0.0


def banner() -> None:
    """Persistent synthetic-data notice shown on every page."""
    st.warning(
        "**SYNTHETIC DEMO** — all data is generated (`synthetic_v1`, seed "
        f"{config.RANDOM_SEED}). Figures are illustrative: not real market performance, "
        "not causal marketing ROI, not clinical guidance.",
        icon="⚠️",
    )


def missing(path: Path, command: str) -> bool:
    """Show a clear instruction and return True if `path` is missing."""
    if not path.exists():
        st.error(f"Missing `{path.relative_to(ROOT)}`. Run `{command}` first.")
        return True
    return False


@st.cache_data(show_spinner=False)
def _read_parquet(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _read_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_parquet(path: Path) -> pd.DataFrame:
    return _read_parquet(str(path), _mtime(path))


def load_csv(path: Path) -> pd.DataFrame:
    return _read_csv(str(path), _mtime(path))


def load_json(path: Path) -> dict:
    return _read_json(str(path), _mtime(path))


def load_panel() -> pd.DataFrame | None:
    """Clean panel with readable names, or None (with an error shown) if not generated."""
    if missing(config.PANEL_PARQUET, "python scripts/bootstrap.py"):
        return None
    df = load_parquet(config.PANEL_PARQUET).copy()
    df["region_name"] = df["region_id"].map(config.REGIONS)
    df["product_name"] = df["product_id"].map(config.PRODUCTS)
    return df


def filters(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Sidebar date / region / product filters; returns the filtered frame."""
    st.sidebar.subheader("Filters")
    months = sorted(df["month_start"].dt.date.unique())
    start, end = st.sidebar.select_slider("Month range", options=months, value=(months[0], months[-1]), key=f"{key}_months")
    regions = st.sidebar.multiselect("Regions", sorted(df["region_name"].unique()), default=sorted(df["region_name"].unique()), key=f"{key}_r")
    products = st.sidebar.multiselect("Products", sorted(df["product_name"].unique()), default=sorted(df["product_name"].unique()), key=f"{key}_p")
    mask = (df["month_start"].dt.date >= start) & (df["month_start"].dt.date <= end) \
        & df["region_name"].isin(regions) & df["product_name"].isin(products)
    return df[mask]


def caption_scope(df: pd.DataFrame) -> None:
    """Row count and time window shown near charts."""
    if len(df):
        st.caption(f"{len(df):,} rows · {df['month_start'].min():%b %Y} – {df['month_start'].max():%b %Y} · units = packs sold, money = INR")
    else:
        st.caption("No rows match the current filters.")
