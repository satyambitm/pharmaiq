"""Ask questions in plain language; answers come only from read-only tools over one dataset."""

from __future__ import annotations

import streamlit as st

from services import api_client
from services.api_client import ApiError
from services.ui import call, page, show_banner

EXAMPLES = {
    "demo": ["Which region had the highest net revenue, and how fast is it growing?",
             "Forecast Therapy B in the South for the next 3 months. How accurate is the model?",
             "What drives the forecasts?",
             "Compare low, base and high scenarios for a 100 crore budget."],
    "upload": ["Summarise sales by region.", "What does the data quality report say?",
               "Forecast the largest region for the next 3 months.", "Which segments did the clustering find?"],
}
MAX_HISTORY = 12

page("AI Business Analyst", banner=None)
status = call(api_client.agent_status, spinner="Checking the AI analyst...")
if not status["enabled"]:
    show_banner("synthetic")
    st.info(status["note"] + " The operator enables it with `AGENT_MODE` and `OPENAI_API_KEY` (see `docs/agent.md`).")
    st.stop()

# ---------------------------------------------------------------- scope: demo or one of your datasets
key: str | None = None
scope_id: str | None = None
scope_label = "Synthetic demo"
uploads = call(api_client.dataset_settings, spinner="Checking uploads...")
with st.sidebar:
    st.subheader("Ask about")
    options = ["Synthetic demo"]
    if uploads["uploads_enabled"] and uploads["store_available"]:
        if uploads["mode"] == "api_key" or status["mode"] == "api_key":
            key = st.text_input("Access key", type="password", key="agent_key",
                                help="Needed for your uploaded datasets" + (" and for the analyst itself"
                                                                            if status["mode"] == "api_key" else ""))
        if key or uploads["mode"] == "local":
            try:
                ready = [d for d in api_client.list_datasets(key)["items"] if d["status"] == "ready"]
            except ApiError as exc:
                st.error(exc.message)
                ready = []
            by_label = {f"{d['dataset_name']} (uploaded)": d["dataset_id"] for d in ready}
            options += list(by_label)
            preferred = next((lbl for lbl, i in by_label.items() if i == st.session_state.get("dataset_id")), None)
            scope_label = st.radio("Dataset", options, index=options.index(preferred) if preferred else 0)
            scope_id = by_label.get(scope_label)
    elif status["mode"] == "api_key":
        key = st.text_input("Access key", type="password", key="agent_key")
    if status["mode"] == "api_key" and not key:
        st.info("Enter your access key to use the analyst.")
    st.caption(f"Model: {status['model']} · up to {status['requests_per_hour']} questions per hour")

show_banner("byod" if scope_id else "synthetic")
if status["mode"] == "api_key" and not key:
    st.stop()
st.caption("Answers use only six read-only tools (sales summary, forecasts, segments, SHAP explanations, budget "
           "scenarios, data quality) over the dataset selected in the sidebar. Every answer shows which tools ran. "
           "The analyst cannot change data or see anyone else's.")

# ---------------------------------------------------------------- conversation (one per dataset)
convs = st.session_state.setdefault("agent_conversations", {})
conv: list[dict] = convs.setdefault(scope_id or "demo", [])


def render(turn: dict) -> None:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        meta = turn.get("meta")
        if not meta:
            return
        if meta["grounding"]["unverified"]:
            st.warning("These numbers were not found in any tool result, so treat them with caution: "
                       + ", ".join(meta["grounding"]["unverified"]))
        for c in meta["caveats"]:
            st.caption(f"ℹ️ {c}")
        with st.expander("How this answer was produced"):
            st.write(f"Dataset: **{meta['dataset']['label']}**"
                     + (f" · model version `{meta['dataset']['model_version'][:8]}`"
                        if meta["dataset"].get("model_version") else ""))
            if meta["tool_calls"]:
                for t in meta["tool_calls"]:
                    st.write(f"{'✅' if t['ok'] else '⚠️'} `{t['tool']}` {t['args'] or ''}")
            else:
                st.write("No tools were called, so this answer contains no figures from the data.")
            st.caption(f"{meta['model']} · {meta['duration_ms'] / 1000:.1f} s · "
                       f"{meta['grounding']['numbers_checked']} numbers checked against tool results")


for turn in conv:
    render(turn)

prompt = st.chat_input("Ask about sales, forecasts, segments, drivers, budget scenarios or data quality")
if not conv and not prompt:
    st.write("Try one of these:")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLES["upload" if scope_id else "demo"]):
        if cols[i % 2].button(q, key=f"ex_{i}", width="stretch"):
            prompt = q

if prompt:
    history = [{"role": t["role"], "content": t["content"][:4000]} for t in conv][-MAX_HISTORY:]
    user_turn = {"role": "user", "content": prompt}
    conv.append(user_turn)
    render(user_turn)
    with st.chat_message("assistant"), st.spinner("Analysing..."):
        try:
            res = api_client.agent_chat(prompt, history, scope_id, key)
            reply = {"role": "assistant", "content": res["answer"], "meta": res}
        except ApiError as exc:
            conv.pop()
            st.error(exc.message)
            st.stop()
    conv.append(reply)
    st.rerun()

if conv and st.sidebar.button("Clear this conversation"):
    convs.pop(scope_id or "demo", None)
    st.rerun()
