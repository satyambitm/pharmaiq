"""API contract tests against the bundled CSV backend and the committed model bundle."""

import pytest

from backend.app.core.config import Settings


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.headers["X-Request-ID"]


def test_ready_reports_data_and_models(client):
    body = client.get("/health/ready").json()
    assert body["status"] == "ready" and body["data_rows"] == 1080
    assert body["data_window"] == "2021-01..2025-12"
    assert set(body["model_versions"]) == {"xgboost", "prophet"}


def test_overview_kpis_reconcile_with_monthly(client):
    body = client.get("/api/v1/analytics/overview").json()
    assert body["meta"]["data_source"] == "synthetic_v1" and "synthetic" in body["meta"]["disclaimer"]
    assert body["months"] == 60 and len(body["monthly"]) == 60
    assert sum(m["units_sold"] for m in body["monthly"]) == body["kpis"]["units_sold"]
    assert sum(p["share_pct"] for p in body["product_mix"]) == pytest.approx(100, abs=0.05)


def test_regions_filters(client):
    body = client.get("/api/v1/analytics/regions",
                      params={"product_id": ["P01", "P02"], "start_month": "2025-01-01", "end_month": "2025-06-01"}).json()
    assert body["product_ids"] == ["P01", "P02"]
    assert body["window_start"] == "2025-01-01" and body["window_end"] == "2025-06-01"
    assert len(body["regions"]) == 6
    assert all(r["growth_pct_last12_vs_prior12"] is None for r in body["regions"])


def test_regions_rejects_reversed_window(client):
    r = client.get("/api/v1/analytics/regions", params={"start_month": "2025-06-01", "end_month": "2025-01-01"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"


@pytest.mark.parametrize("model", ["xgboost", "prophet"])
def test_forecast_shape(client, model):
    body = client.get("/api/v1/forecasts", params={"region_id": "R03", "horizon": 2, "model": model}).json()
    assert body["origin_month"] == "2025-12-01" and body["model"] == model
    assert body["generated"] == ("live" if model == "xgboost" else "precomputed")
    assert len(body["forecast"]) == 3 * 2
    assert {p["target_month"] for p in body["forecast"]} == {"2026-01-01", "2026-02-01"}
    for p in body["forecast"]:
        assert p["lower_units"] <= p["predicted_units"] <= p["upper_units"]
    assert [a["horizon_months"] for a in body["backtest_accuracy_2025"]] == [1, 2]
    assert body["interval"]["level"] == 0.8


def test_xgboost_forecast_is_deterministic(client):
    params = {"region_id": "R01", "horizon": 3}
    assert client.get("/api/v1/forecasts", params=params).json() == client.get("/api/v1/forecasts", params=params).json()


@pytest.mark.parametrize("params", [
    {"region_id": "R99"}, {"region_id": "R01", "horizon": 0}, {"region_id": "R01", "horizon": 4},
    {"region_id": "R01", "model": "arima"}, {},
])
def test_forecast_rejects_unsupported_inputs(client, params):
    r = client.get("/api/v1/forecasts", params=params)
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_segments(client):
    body = client.get("/api/v1/segments").json()
    assert body["k"] == 3 and len(body["cells"]) == 18
    assert sum(p["n_cells"] for p in body["profiles"]) == 18


def test_model_insights_has_caveat_and_scores(client):
    body = client.get("/api/v1/model-insights").json()
    assert "not causal" in body["shap_caveat"]
    assert {(s["model"], s["horizon_months"]) for s in body["horizon_scorecard"]} >= {("xgboost", 1), ("prophet", 3)}
    assert body["global_importance"] and body["local_examples"]


def test_optimizer_conserves_budget_and_labels_scenario(client):
    body = client.post("/api/v1/optimize-budget", json={"total_budget_cr": 120, "scenario": "base"}).json()
    assert "Scenario-based" in body["label"]
    opt = next(s for s in body["strategies"] if s["strategy"] == "optimized")
    assert sum(a["spend_cr"] for a in opt["allocation"]) == pytest.approx(120, abs=0.01)
    assert opt["feasible"] and body["optimized_vs_equal_split_cr"] >= 0
    assert body["feasible_budget_range_cr"] == [30, 180] and len(body["sensitivity"]) == 15


def test_optimizer_custom_bounds(client):
    body = client.post("/api/v1/optimize-budget",
                       json={"total_budget_cr": 60, "min_spend_cr": 8, "max_spend_cr": 12}).json()
    opt = next(s for s in body["strategies"] if s["strategy"] == "optimized")
    assert all(8 - 1e-6 <= a["spend_cr"] <= 12 + 1e-6 for a in opt["allocation"])


@pytest.mark.parametrize("payload,code", [
    ({"total_budget_cr": 10}, "invalid_request"),
    ({"total_budget_cr": -5}, "validation_error"),
    ({"total_budget_cr": 120, "scenario": "extreme"}, "validation_error"),
    ({"total_budget_cr": 120, "competitor_pressure": {"Atlantis": 0.1}}, "validation_error"),
    ({"total_budget_cr": 120, "competitor_pressure": {"North": 1.5}}, "validation_error"),
    ({"total_budget_cr": 120, "min_spend_cr": 40, "max_spend_cr": 30}, "validation_error"),
])
def test_optimizer_rejects_bad_input(client, payload, code):
    r = client.post("/api/v1/optimize-budget", json=payload)
    assert r.status_code == 422 and r.json()["error"]["code"] == code


def test_unhandled_errors_hide_internals(client, monkeypatch):
    from backend.app.services import segmentation_service

    def boom(_):
        raise RuntimeError("secret connection string postgresql://user:pw@host/db")

    monkeypatch.setattr(segmentation_service, "segments", boom)
    from fastapi.testclient import TestClient
    with TestClient(client.app, raise_server_exceptions=False) as c:
        r = c.get("/api/v1/segments")
    assert r.status_code == 500
    assert "secret" not in r.text and r.json()["error"]["code"] == "internal_error"


def test_database_url_normalised_and_hidden():
    s = Settings(data_backend="postgres", database_url="postgres://u:pw@db.example.com:5432/x")
    assert s.sqlalchemy_url.startswith("postgresql+psycopg://")
    assert "pw" not in repr(s)


def test_postgres_requires_url():
    with pytest.raises(ValueError):
        Settings(data_backend="postgres", database_url=None, _env_file=None)
