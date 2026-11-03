"""AI Business Analyst: dataset binding, owner isolation, grounding, caveats, limits. A scripted fake model stands
in for the LLM, so no request leaves the machine."""

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

from backend.app.agent.context import DemoContext
from backend.app.agent.limits import AgentLimiter
from backend.app.agent.runner import grounding
from backend.app.agent.tools import build_tools
from backend.app.core.config import get_settings
from backend.app.db.training import TrainingStore
from backend.app.main import create_app
from ml import byod_worker

ROOT = Path(__file__).resolve().parents[2]
TERRITORY = (ROOT / "data" / "byod_examples" / "territory_monthly_sales.csv").read_bytes()
ALICE, BOB = "alice-test-key-0123456789", "bob-test-key-9876543210"
A, B = {"X-API-Key": ALICE}, {"X-API-Key": BOB}
CHAT = "/api/v1/agent/chat"


class FakeLLM:
    """Replays a script. Each step is an AIMessage or a function of the message list returning one."""

    def __init__(self, *steps):
        self.steps, self.calls, self.tool_names = list(steps), [], []

    def bind_tools(self, tools):
        self.tool_names = [t.name for t in tools]
        self.tool_schemas = {t.name: t.args for t in tools}
        return self

    def invoke(self, msgs):
        self.calls.append(list(msgs))
        step = self.steps.pop(0) if self.steps else AIMessage("done")
        return step(msgs) if callable(step) else step


def call(name, args=None, cid="c1"):
    return AIMessage("", tool_calls=[{"name": name, "args": args or {}, "id": cid}])


def last_tool_result(msgs) -> dict:
    return json.loads(next(m for m in reversed(msgs) if isinstance(m, ToolMessage)).content)


def _digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


@contextmanager
def _client(monkeypatch, tmp_path, **env):
    env = {"AGENT_MODE": "public", "OPENAI_API_KEY": "sk-test-not-a-real-key", "UPLOADS_MODE": "api_key",
           "API_KEYS": f"alice:{_digest(ALICE)},bob:{_digest(BOB)}",
           "DATASET_STORE_URL": f"sqlite:///{tmp_path / 'store.db'}", **env}
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as c:
            c.app.state.agent_llm_factory = lambda settings: pytest.fail("no LLM factory set for this test")
            yield c
    finally:
        get_settings.cache_clear()


@pytest.fixture
def api(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path) as c:
        yield c


def use(c, llm: FakeLLM) -> FakeLLM:
    c.app.state.agent_llm_factory = lambda settings: llm
    return llm


def _ready_dataset(c, headers=A) -> str:
    up = c.post("/api/v1/datasets/upload", files={"file": ("t.csv", TERRITORY, "text/csv")}, headers=headers).json()
    did = up["dataset"]["dataset_id"]
    c.post(f"/api/v1/datasets/{did}/validate", json={"mapping": up["profile"]["suggested_mapping"]}, headers=headers)
    assert c.post(f"/api/v1/datasets/{did}/activate", headers=headers).status_code == 200
    return did


def test_disabled_without_mode_or_key(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path, AGENT_MODE="disabled") as c:
        assert c.get("/api/v1/agent/status").json()["enabled"] is False
        r = c.post(CHAT, json={"message": "hi"})
        assert r.status_code == 503 and r.json()["error"]["code"] == "agent_disabled"


def test_no_tool_accepts_a_dataset_id():
    tools = build_tools(DemoContext(repo=None, bundle=None))
    assert [t.name for t in tools] == ["get_sales_summary", "get_sales_forecast", "get_market_segments",
                                       "explain_forecast", "compare_budget_scenarios", "get_data_quality"]
    for t in tools:
        assert not {"dataset_id", "owner_id", "dataset", "sql", "path"} & set(t.args)


def test_demo_answer_is_grounded_in_tool_results(api):
    def answer(msgs):
        kpis = last_tool_result(msgs)["kpis"]
        return AIMessage(f"Net revenue was ₹{kpis['net_revenue_cr']} cr; units {kpis['units_sold']:,}. "
                         "Next year it will reach 999.75 cr.")
    llm = use(api, FakeLLM(call("get_sales_summary"), answer))
    r = api.post(CHAT, json={"message": "How did sales go overall?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dataset"]["kind"] == "demo" and body["tool_calls"] == [{"tool": "get_sales_summary", "args": {},
                                                                         "ok": True}]
    assert body["grounding"]["unverified"] == ["999.75"] and body["grounding"]["numbers_checked"] == 3
    system = llm.calls[0][0].content
    assert "Synthetic demo dataset" in system and "Never estimate" in system


def test_shap_and_budget_caveats_are_returned(api):
    use(api, FakeLLM(call("explain_forecast"), call("compare_budget_scenarios", {"total_budget_cr": 100}, "c2"),
                     AIMessage("Lag features matter most.")))
    body = api.post(CHAT, json={"message": "What drives the forecast and how should I split 100 cr?"}).json()
    assert [c["tool"] for c in body["tool_calls"]] == ["explain_forecast", "compare_budget_scenarios"]
    text = " ".join(body["caveats"]).lower()
    assert "not a measured roi" in text and ("caus" in text or "association" in text)


def test_tool_errors_go_back_to_the_model(api):
    def check(msgs):
        assert "Unknown region" in last_tool_result(msgs)["error"]
        return call("drop_tables", {}, "c2")

    def check_unknown(msgs):
        assert "Unknown tool" in last_tool_result(msgs)["error"]
        return AIMessage("There is no region called Atlantis.")
    use(api, FakeLLM(call("get_sales_forecast", {"region": "Atlantis"}), check, check_unknown))
    body = api.post(CHAT, json={"message": "Forecast Atlantis"}).json()
    assert [c["ok"] for c in body["tool_calls"]] == [False, False]


def test_tool_call_cap(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path, AGENT_MAX_TOOL_CALLS="2") as c:
        use(c, FakeLLM(*[call("get_data_quality", cid=f"c{i}") for i in range(10)]))
        body = c.post(CHAT, json={"message": "loop"}).json()
        assert [x["ok"] for x in body["tool_calls"]] == [True, True, False]
        assert "could not finish" in body["answer"]


def test_uploaded_dataset_is_owner_scoped_and_limits_are_explained(api):
    did = _ready_dataset(api)
    assert api.post(CHAT, json={"message": "hi", "dataset_id": did}).status_code == 401
    assert api.post(CHAT, json={"message": "hi", "dataset_id": did}, headers=B).status_code == 404

    def budget_then_answer(msgs):
        assert last_tool_result(msgs)["available"] is False
        return call("get_sales_forecast", {"region": "North"}, "c2")
    use(api, FakeLLM(call("compare_budget_scenarios", {"total_budget_cr": 10}), budget_then_answer,
                     lambda msgs: AIMessage(json.dumps(last_tool_result(msgs)))))
    body = api.post(CHAT, json={"message": "Plan budget, then forecast North", "dataset_id": did}, headers=A).json()
    ref = body["dataset"]
    assert (ref["kind"], ref["dataset_id"], ref["model_version"]) == ("upload", did, None)
    assert [c["ok"] for c in body["tool_calls"]] == [True, True]
    assert "No models have been trained" in body["answer"]


def test_uploaded_dataset_after_training(api):
    did = _ready_dataset(api)
    api.post("/api/v1/models/train", json={"dataset_id": did}, headers=A)
    datasets = api.app.state.dataset_store
    training = TrainingStore(datasets.engine)
    version_id = byod_worker.process(training.claim("t"), datasets, training, with_prophet=False)

    seen = {}

    def answer(msgs):
        seen["forecast"] = last_tool_result(msgs)
        return call("get_sales_summary", {}, "c2")

    def summary(msgs):
        seen["summary"] = last_tool_result(msgs)
        return AIMessage("ok")
    use(api, FakeLLM(call("get_sales_forecast", {"region": "chennai", "horizon": 2}), answer, summary))
    body = api.post(CHAT, json={"message": "Forecast Chennai", "dataset_id": did}, headers=A).json()
    assert body["dataset"]["model_version"] == version_id
    fc = seen["forecast"]
    assert fc["model"] == fc["recommended_model"] and fc["forecast"] and {r["step"] for r in fc["forecast"]} <= {1, 2}
    assert "holdout_evaluation" in fc
    s = seen["summary"]
    assert s["totals"]["units_sold"] > 0 and "revenue" in s["totals"]  # the sample has prices


def test_rate_limit(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path, AGENT_REQUESTS_PER_HOUR="2") as c:
        use(c, FakeLLM(*[AIMessage("hi")] * 5))
        assert [c.post(CHAT, json={"message": "q"}).status_code for _ in range(3)] == [200, 200, 429]


def test_api_key_mode_requires_a_key(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path, AGENT_MODE="api_key") as c:
        use(c, FakeLLM(AIMessage("hi")))
        assert c.post(CHAT, json={"message": "q"}).status_code == 401
        assert c.post(CHAT, json={"message": "q"}, headers=A).status_code == 200


def test_provider_errors_are_not_echoed(api):
    class Boom(FakeLLM):
        def invoke(self, msgs):
            raise RuntimeError("401 invalid key sk-secret-value")
    use(api, Boom())
    r = api.post(CHAT, json={"message": "q"})
    assert r.status_code == 502 and "sk-secret" not in r.text


def test_request_validation(api):
    assert api.post(CHAT, json={"message": ""}).status_code == 422
    assert api.post(CHAT, json={"message": "x" * 2001}).status_code == 422
    assert api.post(CHAT, json={"message": "q", "history": [{"role": "system", "content": "x"}]}).status_code == 422
    assert api.post(CHAT, json={"message": "q", "dataset_id": "../etc"}).status_code == 422


def test_grounding_rules():
    src = {484.1, 0.0623, 6.23, 4841000000.0}
    assert grounding("Revenue ₹484.1 cr, WAPE 6.2%, growth 6.23%", src)["unverified"] == []
    assert grounding("Share 6.2% of 484.10", src)["unverified"] == []
    assert grounding("It will grow 37.5% to 1,234", src)["unverified"] == ["37.5", "1234"]
    assert grounding("Top 3 regions over 6 months", src)["numbers_checked"] == 0


def test_limiter_windows():
    lim = AgentLimiter(per_hour=2, per_day=3)
    assert [lim.check("a", 0), lim.check("a", 1), lim.check("a", 2) is not None] == [None, None, True]
    assert lim.check("a", 3_700) is None
    assert lim.check("b", 3_701) is not None  # daily cap reached across clients
