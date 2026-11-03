"""Shared page furniture: synthetic-data banner, backend status and safe API calls."""

from __future__ import annotations

from typing import Callable

import streamlit as st

from services import api_client
from services.api_client import ApiError

REGIONS = {"R01": "North", "R02": "South", "R03": "East", "R04": "West", "R05": "Central", "R06": "Northeast"}
PRODUCTS = {"P01": "Therapy A", "P02": "Therapy B", "P03": "Therapy C"}


def page(title: str, banner: str = "synthetic") -> None:
    st.set_page_config(page_title=f"PharmaIQ · {title}", page_icon="💊", layout="wide")
    st.title(title)
    if banner == "synthetic":
        st.warning("**Synthetic demo**: every number comes from generated data (`synthetic_v1`, seed 42). "
                   "Results are illustrative. They are not real market performance, causal marketing ROI or clinical "
                   "guidance.", icon="⚠️")
    else:
        st.info("**Your data**: this page works on datasets you upload. They are stored separately from the synthetic "
                "demo, which is never modified. Other pages still show the synthetic demo.", icon="📁")
    _sidebar_status()


def _sidebar_status() -> None:
    url = api_client.base_url()
    with st.sidebar:
        st.caption(f"API: `{url}`")
        ready = api_client.readiness()
        if ready.get("status") == "ready":
            st.success(f"Backend ready · {ready.get('data_backend')} · {ready.get('data_window')}", icon="✅")
        else:
            st.error("Backend not ready: " + "; ".join(ready.get("problems") or ["unknown"]), icon="🛑")
            if st.button("Retry connection"):
                api_client.readiness.clear()
                st.rerun()
        if "localhost" in url or "127.0.0.1" in url:
            st.caption("Using the local development API. Set `API_BASE_URL` for deployments.")


def call(fn: Callable, *args, spinner: str = "Loading from the API...", **kwargs) -> dict:
    """Run an API call with a spinner; on failure show the error and stop the page."""
    try:
        with st.spinner(spinner):
            return fn(*args, **kwargs)
    except ApiError as exc:
        if exc.status is None:
            api_client.readiness.clear()
        st.error(exc.message)
        if exc.details:
            st.json(exc.details, expanded=False)
        st.stop()
