"""Thin HTTP client for the PharmaIQ API, shared by every Streamlit page.

The base URL comes from the `API_BASE_URL` environment variable or Streamlit secret.
The localhost default is for local development only.
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOCAL_DEFAULT = "http://localhost:8000"
CONNECT_TIMEOUT = 10
# Render's free tier sleeps when idle and can take about a minute to wake on the first request.
READ_TIMEOUT = 90
CACHE_TTL = 600


class ApiError(Exception):
    """A failure with a message that is safe and useful to show in the UI."""

    def __init__(self, message: str, status: int | None = None, details: Any = None) -> None:
        super().__init__(message)
        self.message, self.status, self.details = message, status, details


def base_url() -> str:
    url = os.getenv("API_BASE_URL")
    if not url:
        try:
            url = st.secrets.get("API_BASE_URL")
        except Exception:  # Streamlit raises when no secrets file exists
            url = None
    return (url or LOCAL_DEFAULT).rstrip("/")


@st.cache_resource
def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=2, connect=2, read=0, status=2, backoff_factor=1.0,
                  status_forcelist=(502, 503, 504), allowed_methods=frozenset({"GET"}))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["Accept"] = "application/json"
    return s


def _request(method: str, path: str, **kwargs) -> dict:
    url = f"{base_url()}{path}"
    try:
        r = _session().request(method, url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), **kwargs)
    except requests.exceptions.ConnectTimeout as exc:
        raise ApiError("The API did not respond in time. If it is hosted on a free tier it may be waking up; "
                       "wait a minute and retry.") from exc
    except requests.exceptions.ConnectionError as exc:
        raise ApiError(f"Cannot reach the API at {base_url()}. Check that the backend is running and "
                       "API_BASE_URL is correct.") from exc
    except requests.exceptions.ReadTimeout as exc:
        raise ApiError("The API took too long to answer. It may be waking up from sleep; please retry.") from exc
    except requests.exceptions.RequestException as exc:
        raise ApiError("The request to the API failed.") from exc

    if not r.content:
        raise ApiError(f"The API returned an empty response (HTTP {r.status_code}).", r.status_code)
    try:
        body = r.json()
    except ValueError as exc:
        raise ApiError(f"The API returned a non-JSON response (HTTP {r.status_code}).", r.status_code) from exc
    if r.status_code >= 400:
        err = body.get("error", {}) if isinstance(body, dict) else {}
        msg = err.get("message") or f"API error (HTTP {r.status_code})"
        if err.get("request_id"):
            msg += f" [request id {err['request_id']}]"
        raise ApiError(msg, r.status_code, err.get("details"))
    return body


@st.cache_data(ttl=60, show_spinner=False)
def readiness() -> dict:
    """Readiness never raises: a 503 body still describes what is wrong."""
    try:
        return _request("GET", "/health/ready")
    except ApiError as exc:
        return {"status": "unreachable", "problems": [exc.message]}


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def overview(start_month: str | None = None, end_month: str | None = None) -> dict:
    return _request("GET", "/api/v1/analytics/overview", params={"start_month": start_month, "end_month": end_month})


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def regions(start_month: str | None = None, end_month: str | None = None,
            product_ids: tuple[str, ...] = ()) -> dict:
    params = {"start_month": start_month, "end_month": end_month, "product_id": list(product_ids) or None}
    return _request("GET", "/api/v1/analytics/regions", params=params)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def forecast(region_id: str, horizon: int, model: str) -> dict:
    return _request("GET", "/api/v1/forecasts", params={"region_id": region_id, "horizon": horizon, "model": model})


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def segments() -> dict:
    return _request("GET", "/api/v1/segments")


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def model_insights() -> dict:
    return _request("GET", "/api/v1/model-insights")


# ---------------------------------------------------------------- user datasets
# Never cached: responses are private to the caller's access key and change as the user works.

def _auth(api_key: str | None) -> dict:
    return {"X-API-Key": api_key} if api_key else {}


def dataset_settings() -> dict:
    return _request("GET", "/api/v1/datasets/settings")


def upload_dataset(api_key: str | None, file_name: str, content: bytes, dataset_name: str | None) -> dict:
    return _request("POST", "/api/v1/datasets/upload", headers=_auth(api_key),
                    files={"file": (file_name, content, "application/octet-stream")},
                    data={"dataset_name": dataset_name or ""})


def list_datasets(api_key: str | None) -> dict:
    return _request("GET", "/api/v1/datasets", headers=_auth(api_key), params={"limit": 100})


def dataset_detail(api_key: str | None, dataset_id: str) -> dict:
    return _request("GET", f"/api/v1/datasets/{dataset_id}", headers=_auth(api_key))


def dataset_profile(api_key: str | None, dataset_id: str) -> dict:
    return _request("GET", f"/api/v1/datasets/{dataset_id}/profile", headers=_auth(api_key))


def validate_dataset(api_key: str | None, dataset_id: str, body: dict) -> dict:
    return _request("POST", f"/api/v1/datasets/{dataset_id}/validate", headers=_auth(api_key), json=body)


def activate_dataset(api_key: str | None, dataset_id: str) -> dict:
    return _request("POST", f"/api/v1/datasets/{dataset_id}/activate", headers=_auth(api_key))


def delete_dataset(api_key: str | None, dataset_id: str) -> dict:
    return _request("DELETE", f"/api/v1/datasets/{dataset_id}", headers=_auth(api_key))


def agent_status() -> dict:
    return _request("GET", "/api/v1/agent/status")


def agent_chat(message: str, history: list[dict], dataset_id: str | None, api_key: str | None) -> dict:
    return _request("POST", "/api/v1/agent/chat", headers=_auth(api_key),
                    json={"message": message, "history": history, "dataset_id": dataset_id})


def train_models(api_key: str | None, dataset_id: str) -> dict:
    return _request("POST", "/api/v1/models/train", headers=_auth(api_key), json={"dataset_id": dataset_id})


def training_jobs(api_key: str | None, dataset_id: str) -> dict:
    return _request("GET", "/api/v1/models/jobs", headers=_auth(api_key), params={"dataset_id": dataset_id})


def model_versions(api_key: str | None, dataset_id: str) -> dict:
    return _request("GET", "/api/v1/models/versions", headers=_auth(api_key), params={"dataset_id": dataset_id})


def model_version(api_key: str | None, version_id: str) -> dict:
    return _request("GET", f"/api/v1/models/versions/{version_id}", headers=_auth(api_key))


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def optimize_budget(total_budget_cr: float, scenario: str, pressure: tuple[tuple[str, float], ...],
                    min_spend_cr: float, max_spend_cr: float) -> dict:
    payload = {"total_budget_cr": total_budget_cr, "scenario": scenario, "competitor_pressure": dict(pressure),
               "min_spend_cr": min_spend_cr, "max_spend_cr": max_spend_cr}
    return _request("POST", "/api/v1/optimize-budget", json=payload)
