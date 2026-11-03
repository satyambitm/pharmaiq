"""Dataset upload API: authentication, owner isolation, limits and the upload -> validate -> activate flow.

Storage is a throwaway SQLite file per client; the demo `client` fixture (uploads disabled) is untouched.
"""

import hashlib
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings, get_settings
from backend.app.main import create_app

ROOT = Path(__file__).resolve().parents[2]
TERRITORY = (ROOT / "data" / "byod_examples" / "territory_monthly_sales.csv").read_bytes()
ALICE, BOB = "alice-test-key-0123456789", "bob-test-key-9876543210"
BASE = "/api/v1/datasets"


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


def _upload(c, key, content=TERRITORY, name="territory.csv"):
    return c.post(f"{BASE}/upload", files={"file": (name, content, "text/csv")}, data={"dataset_name": "Territory"},
                  headers={"X-API-Key": key})


def test_uploads_disabled_by_default(client):
    s = client.get(f"{BASE}/settings").json()
    assert s["uploads_enabled"] is False and s["store_available"] is False
    r = _upload(client, ALICE)
    assert r.status_code == 403 and r.json()["error"]["code"] == "uploads_disabled"


def test_local_mode_refused_in_production():
    with pytest.raises(ValueError, match="not allowed"):
        Settings(_env_file=None, environment="production", uploads_mode="local")


def test_malformed_api_keys_refused():
    with pytest.raises(ValueError):
        Settings(_env_file=None, uploads_mode="api_key", api_keys="alice:not-a-hash")


def test_missing_or_wrong_key_is_401(api):
    assert api.get(BASE).status_code == 401
    assert api.get(BASE, headers={"X-API-Key": "wrong"}).status_code == 401
    assert _upload(api, "wrong").status_code == 401


def test_full_flow_and_owner_isolation(api):
    a, b = {"X-API-Key": ALICE}, {"X-API-Key": BOB}
    up = _upload(api, ALICE)
    assert up.status_code == 201
    body = up.json()
    did = body["dataset"]["dataset_id"]
    assert body["dataset"]["status"] == "uploaded" and body["dataset"]["source_row_count"] == 361
    mapping = body["profile"]["suggested_mapping"]
    assert mapping["units_sold"] == "sales_quantity" and mapping["date"] == "transaction_month"

    # Bob can neither see nor touch Alice's dataset, and cannot learn that it exists.
    assert api.get(f"{BASE}/{did}/profile", headers=a).json()["profile"]["suggested_mapping"] == mapping
    assert api.get(BASE, headers=b).json()["total"] == 0
    for method, path in (("get", f"{BASE}/{did}"), ("get", f"{BASE}/{did}/profile"), ("delete", f"{BASE}/{did}"),
                         ("post", f"{BASE}/{did}/activate")):
        assert getattr(api, method)(path, headers=b).status_code == 404
    assert api.post(f"{BASE}/{did}/validate", json={"mapping": mapping}, headers=b).status_code == 404

    assert api.post(f"{BASE}/{did}/activate", headers=a).status_code == 409  # not validated yet

    v = api.post(f"{BASE}/{did}/validate", json={"mapping": mapping}, headers=a)
    assert v.status_code == 200
    rep = v.json()["report"]
    assert rep["status"] == "ready_to_ingest" and rep["valid_rows"] == 359 and rep["rejected_rows"] == 2
    assert rep["rejection_reasons"] == {"invalid_date": 1, "negative_units_sold": 1}
    assert v.json()["dataset"]["status"] == "validated" and v.json()["monthly_preview"]

    act = api.post(f"{BASE}/{did}/activate", headers=a)
    assert act.status_code == 200
    meta = act.json()["dataset"]
    assert meta["status"] == "ready" and meta["row_count"] > 0 and meta["date_start"] <= meta["date_end"]
    assert api.post(f"{BASE}/{did}/activate", headers=a).status_code == 200  # idempotent
    assert api.post(f"{BASE}/{did}/validate", json={"mapping": mapping}, headers=a).status_code == 409

    detail = api.get(f"{BASE}/{did}", headers=a).json()
    assert detail["report"]["valid_rows"] == 359 and detail["dataset"]["schema_mapping"]["columns"] == mapping
    assert api.get(BASE, headers=a).json()["total"] == 1

    assert api.delete(f"{BASE}/{did}", headers=a).json() == {"dataset_id": did, "deleted": True}
    assert api.get(f"{BASE}/{did}", headers=a).status_code == 404


def test_invalid_mapping_is_422_and_nothing_ingested(api):
    a = {"X-API-Key": ALICE}
    did = _upload(api, ALICE).json()["dataset"]["dataset_id"]
    r = api.post(f"{BASE}/{did}/validate", json={"mapping": {"date": "transaction_month", "region": "territory"}},
                 headers=a)
    assert r.status_code == 422 and "Units sold" in r.json()["error"]["message"]
    r = api.post(f"{BASE}/{did}/validate", json={"mapping": {"date": "nope", "region": "territory",
                                                             "units_sold": "sales_quantity", "product": "drug_name"}},
                 headers=a)
    assert r.status_code == 422
    assert api.post(f"{BASE}/{did}/validate", json={"mapping": {"price": "x"}}, headers=a).status_code == 422
    assert api.get(f"{BASE}/{did}", headers=a).json()["dataset"]["status"] == "uploaded"


def test_blocking_report_marks_invalid_and_activation_refused(api):
    a = {"X-API-Key": ALICE}
    bad = b"month,region,units\n" + b"".join(b"2024-%02d,North,-5\n" % m for m in range(1, 13))
    did = _upload(api, ALICE, bad, "bad.csv").json()["dataset"]["dataset_id"]
    r = api.post(f"{BASE}/{did}/validate", headers=a, json={
        "mapping": {"date": "month", "region": "region", "units_sold": "units"}, "single_product_label": "Brand X"})
    assert r.status_code == 200 and r.json()["report"]["status"] == "invalid" and r.json()["report"]["errors"]
    assert r.json()["dataset"]["status"] == "invalid"
    assert api.post(f"{BASE}/{did}/activate", headers=a).status_code == 409


def test_unsafe_and_wrong_type_files(api):
    assert _upload(api, ALICE, b"MZ\x90\x00\x03\x00\x00\x00binary", "sales.csv").status_code == 422
    assert _upload(api, ALICE, b"a,b\n1,2\n", "sales.txt").status_code == 415
    assert _upload(api, ALICE, b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64, "sales.xls").status_code == 415
    assert _upload(api, ALICE, b"", "empty.csv").status_code == 422
    assert api.get(BASE, headers={"X-API-Key": ALICE}).json()["total"] == 0


def test_malformed_dataset_id_is_422(api):
    assert api.get(f"{BASE}/not-a-uuid", headers={"X-API-Key": ALICE}).status_code == 422


def test_size_and_count_limits(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path, MAX_UPLOAD_MB="0.05", MAX_DATASETS_PER_OWNER="1") as c:
        big = b"month,region,units\n" + b"2024-01,North,1\n" * 4000  # ~64 KB: over the file limit, under the body cap
        r = _upload(c, ALICE, big, "big.csv")
        assert r.status_code == 422 and "MB" in r.json()["error"]["message"]
        huge = b"month,region,units\n" + b"2024-01,North,1\n" * 80_000  # ~1.3 MB: rejected before parsing
        r = _upload(c, ALICE, huge, "huge.csv")
        assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"
        small = b"month,region,units\n2024-01,North,1\n"
        assert _upload(c, ALICE, small, "a.csv").status_code == 201
        assert _upload(c, ALICE, small, "b.csv").status_code == 409
        assert _upload(c, BOB, small, "b.csv").status_code == 201  # the limit is per owner


def test_demo_endpoints_unaffected_when_uploads_enabled(api):
    assert api.get("/api/v1/analytics/overview").status_code == 200
    assert api.get("/health/ready").status_code == 200
    assert api.get(f"{BASE}/settings").json()["store_available"] is True
