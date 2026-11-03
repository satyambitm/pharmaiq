"""Training queue and model versions: owner isolation, idempotency, rate limit, and a real worker run."""

import hashlib
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.db.training import TrainingStore
from backend.app.main import create_app
from ml import byod_worker

ROOT = Path(__file__).resolve().parents[2]
TERRITORY = (ROOT / "data" / "byod_examples" / "territory_monthly_sales.csv").read_bytes()
ALICE, BOB = "alice-test-key-0123456789", "bob-test-key-9876543210"
A, B = {"X-API-Key": ALICE}, {"X-API-Key": BOB}
DS, MODELS = "/api/v1/datasets", "/api/v1/models"


def _digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


@contextmanager
def _client(monkeypatch, tmp_path, **env):
    env = {"UPLOADS_MODE": "api_key", "API_KEYS": f"alice:{_digest(ALICE)},bob:{_digest(BOB)}",
           "DATASET_STORE_URL": f"sqlite:///{tmp_path / 'store.db'}", **env}
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as c:
            yield c
    finally:
        get_settings.cache_clear()


@pytest.fixture
def api(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path) as c:
        yield c


def _ready_dataset(c, headers=A) -> str:
    up = c.post(f"{DS}/upload", files={"file": ("t.csv", TERRITORY, "text/csv")}, headers=headers).json()
    did = up["dataset"]["dataset_id"]
    assert c.post(f"{DS}/{did}/validate", json={"mapping": up["profile"]["suggested_mapping"]},
                  headers=headers).status_code == 200
    assert c.post(f"{DS}/{did}/activate", headers=headers).status_code == 200
    return did


def _stores(c):
    datasets = c.app.state.dataset_store
    return datasets, TrainingStore(datasets.engine)


def test_training_requires_auth_and_ready_dataset(api):
    assert api.post(f"{MODELS}/train", json={"dataset_id": "6f1c1b8e-8c1f-4f57-9d55-1f0c7a1d2e3f"}).status_code == 401
    up = api.post(f"{DS}/upload", files={"file": ("t.csv", TERRITORY, "text/csv")}, headers=A).json()
    did = up["dataset"]["dataset_id"]
    r = api.post(f"{MODELS}/train", json={"dataset_id": did}, headers=A)
    assert r.status_code == 409 and "ingestion" in r.json()["error"]["message"]
    assert api.post(f"{MODELS}/train", json={"dataset_id": "nope"}, headers=A).status_code == 422
    assert api.post(f"{MODELS}/train", json={"dataset_id": did, "owner_id": "bob"}, headers=A).status_code == 422


def test_queue_is_idempotent_and_owner_scoped(api):
    did = _ready_dataset(api)
    first = api.post(f"{MODELS}/train", json={"dataset_id": did}, headers=A)
    assert first.status_code == 202 and first.json()["created"] is True
    job = first.json()["job"]
    assert job["status"] == "queued" and job["dataset_id"] == did
    again = api.post(f"{MODELS}/train", json={"dataset_id": did}, headers=A).json()
    assert again["created"] is False and again["job"]["job_id"] == job["job_id"]

    assert api.post(f"{MODELS}/train", json={"dataset_id": did}, headers=B).status_code == 404
    assert api.get(f"{MODELS}/jobs/{job['job_id']}", headers=B).status_code == 404
    assert api.get(f"{MODELS}/jobs", headers=B).json()["items"] == []
    assert [j["job_id"] for j in api.get(f"{MODELS}/jobs", params={"dataset_id": did}, headers=A).json()["items"]] \
        == [job["job_id"]]


def test_claim_is_exclusive_and_stale_jobs_are_recovered(api):
    did = _ready_dataset(api)
    job_id = api.post(f"{MODELS}/train", json={"dataset_id": did}, headers=A).json()["job"]["job_id"]
    _, training = _stores(api)
    claimed = training.claim("w1")
    assert claimed["job_id"] == job_id and claimed["status"] == "running" and claimed["attempts"] == 1
    assert training.claim("w2") is None
    assert training.recover_stale(timedelta(minutes=10), 3) == 0
    assert training.recover_stale(timedelta(seconds=-1), 3) == 1  # silence beyond the limit: requeued
    assert training.claim("w2")["attempts"] == 2
    assert training.recover_stale(timedelta(seconds=-1), 2) == 1  # out of attempts: failed
    assert api.get(f"{MODELS}/jobs/{job_id}", headers=A).json()["status"] == "failed"


def test_daily_rate_limit(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path, MAX_TRAINING_JOBS_PER_DAY="1") as c:
        d1, d2 = _ready_dataset(c), _ready_dataset(c)
        assert c.post(f"{MODELS}/train", json={"dataset_id": d1}, headers=A).status_code == 202
        r = c.post(f"{MODELS}/train", json={"dataset_id": d2}, headers=A)
        assert r.status_code == 429 and r.json()["error"]["code"] == "rate_limited"


def test_worker_trains_and_versions_are_served(api):
    did = _ready_dataset(api)
    job_id = api.post(f"{MODELS}/train", json={"dataset_id": did}, headers=A).json()["job"]["job_id"]
    datasets, training = _stores(api)
    job = training.claim("test-worker")
    version_id = byod_worker.process(job, datasets, training, with_prophet=False)
    assert version_id

    status = api.get(f"{MODELS}/jobs/{job_id}", headers=A).json()
    assert status["status"] == "succeeded" and status["version_id"] == version_id and status["error"] is None
    items = api.get(f"{MODELS}/versions", params={"dataset_id": did}, headers=A).json()["items"]
    assert [v["version_id"] for v in items] == [version_id]
    summary = items[0]["summary"]
    assert summary["recommended_model"] in ("xgboost", "naive_last_month")
    assert summary["metrics"]["naive_last_month"]["n"] > 0 and summary["shap"] == "ok"

    detail = api.get(f"{MODELS}/versions/{version_id}", headers=A).json()
    res = detail["results"]
    assert res["forecasting"]["forecasts"] and "not causal" in res["explainability"]["caveat"]
    assert res["prophet"]["status"] == "skipped" and "artifact" not in detail

    # Same data and code: retraining is refused instead of repeating the work.
    r = api.post(f"{MODELS}/train", json={"dataset_id": did}, headers=A)
    assert r.status_code == 409 and version_id in r.json()["error"]["message"]
    assert api.get(f"{MODELS}/versions/{version_id}", headers=B).status_code == 404
    assert api.get(f"{MODELS}/versions", headers=B).json()["items"] == []

    assert api.delete(f"{DS}/{did}", headers=A).status_code == 200
    assert api.get(f"{MODELS}/versions/{version_id}", headers=A).status_code == 404


def test_worker_refuses_job_when_dataset_changed(api):
    did = _ready_dataset(api)
    job_id = api.post(f"{MODELS}/train", json={"dataset_id": did}, headers=A).json()["job"]["job_id"]
    datasets, training = _stores(api)
    job = training.claim("test-worker")
    job["fingerprint"] = "0" * 64
    assert byod_worker.process(job, datasets, training, with_prophet=False) is None
    status = api.get(f"{MODELS}/jobs/{job_id}", headers=A).json()
    assert status["status"] == "failed" and "changed" in status["error"]


def test_models_endpoints_unavailable_when_uploads_disabled(client):
    r = client.get(f"{MODELS}/versions", headers=A)
    assert r.status_code in (403, 503)
