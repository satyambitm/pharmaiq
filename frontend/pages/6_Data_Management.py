"""Upload your own sales file, map its columns, review the quality report and confirm ingestion."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from services import api_client
from services.ui import call, page

NOT_MAPPED = "— not mapped —"
STATUS_LABELS = {"uploaded": "Uploaded · needs mapping", "invalid": "Validation found blocking errors",
                 "validated": "Validated · waiting for your confirmation", "ready": "Ingested · ready"}

page("Data Management", banner="byod")
cfg = call(api_client.dataset_settings, spinner="Checking upload settings...")

if not cfg["uploads_enabled"]:
    st.info(cfg["note"])
    st.markdown("Uploads are turned on by the operator with `UPLOADS_MODE`. See `docs/byod.md` in the repository. "
                "These are the fields a file can be mapped to:")
    st.dataframe(pd.DataFrame(cfg["fields"]), hide_index=True, width="stretch")
    st.stop()
if not cfg["store_available"]:
    st.error("Uploads are enabled but dataset storage is unavailable on the server. Try again later.")
    st.stop()

st.caption(cfg["note"])
key: str | None = None
if cfg["mode"] == "api_key":
    key = st.text_input("Access key", type="password", key="byod_key",
                        help="Issued by the operator. It identifies you; your datasets are visible only with this key.")
    if not key:
        st.info("Enter your access key to upload or view your datasets.")
        st.stop()

# ---------------------------------------------------------------- upload
with st.expander("Upload a new file", expanded=False):
    st.write(f"CSV or Excel (.xlsx), up to {cfg['max_upload_mb']:g} MB and {cfg['max_upload_rows']:,} rows. "
             "Required: a date, a region and units sold; product is required unless the file is single-product. "
             "The original file is stored unchanged.")
    up = st.file_uploader("File", type=["csv", "xlsx"], accept_multiple_files=False)
    name = st.text_input("Dataset name", max_chars=120, placeholder="e.g. FY24 territory sales")
    if st.button("Upload", type="primary", disabled=up is None):
        res = call(api_client.upload_dataset, key, up.name, up.getvalue(), name or None, spinner="Uploading and profiling...")
        st.session_state["byod_selected"] = res["dataset"]["dataset_id"]
        st.session_state.setdefault("byod_profiles", {})[res["dataset"]["dataset_id"]] = res["profile"]
        st.success(f"Uploaded **{res['dataset']['dataset_name']}** ({res['dataset']['source_row_count']:,} rows).")

# ---------------------------------------------------------------- dataset list
listing = call(api_client.list_datasets, key, spinner="Loading your datasets...")
items = listing["items"]
st.subheader(f"Your datasets ({listing['total']} of {cfg['max_datasets_per_owner']} allowed)")
if not items:
    st.info("No datasets yet. Upload a file above.")
    st.stop()

table = pd.DataFrame(items)[["dataset_name", "status", "source_type", "source_row_count", "row_count", "date_start",
                             "date_end", "created_at"]]
st.dataframe(table.rename(columns={"source_row_count": "file rows", "row_count": "monthly rows"}), hide_index=True,
             width="stretch")

ids = [d["dataset_id"] for d in items]
by_id = {d["dataset_id"]: d for d in items}
selected = st.selectbox("Work with", ids, index=ids.index(st.session_state["byod_selected"])
                        if st.session_state.get("byod_selected") in ids else 0,
                        format_func=lambda i: f"{by_id[i]['dataset_name']} · {STATUS_LABELS[by_id[i]['status']]}")
st.session_state["byod_selected"] = selected
meta = by_id[selected]

with st.sidebar:
    st.subheader("Selected dataset")
    st.write(f"**{meta['dataset_name']}**")
    st.caption(STATUS_LABELS[meta["status"]])
    with st.popover("Delete this dataset"):
        st.write("This removes the stored file, the mapping, the report and any ingested rows. It cannot be undone.")
        if st.checkbox("I understand", key=f"del_ok_{selected}") and st.button("Delete permanently", type="primary"):
            call(api_client.delete_dataset, key, selected, spinner="Deleting...")
            st.session_state.pop("byod_selected", None)
            st.session_state.get("byod_profiles", {}).pop(selected, None)
            st.rerun()


def show_report(report: dict, monthly_preview: list[dict] | None = None) -> None:
    ok = report["status"] == "ready_to_ingest"
    (st.success if ok else st.error)("No blocking errors." if ok else "Blocking errors: fix the mapping or the file.")
    for e in report["errors"]:
        st.error(e)
    c = st.columns(4)
    c[0].metric("File rows", f"{report['total_rows']:,}")
    c[1].metric("Valid rows", f"{report['valid_rows']:,}")
    c[2].metric("Rejected rows", f"{report['rejected_rows']:,}")
    c[3].metric("Monthly rows", f"{report['aggregation']['monthly_rows']:,}")
    st.caption(f"Detected source frequency: **{report['detected_frequency']}**. Units, revenue, marketing spend and "
               "doctor interactions are summed per month; price is revenue ÷ units; other fields are monthly averages.")
    for w in report["warnings"]:
        st.warning(w)

    left, right = st.columns(2)
    with left:
        st.markdown("**Coverage**")
        cov = report["coverage"]
        st.write(f"{cov['date_start']} → {cov['date_end']} ({cov['months_spanned']} months) · {cov['n_regions']} regions · "
                 f"{cov['n_products']} products · {cov['n_series']} region/product series")
        st.write(f"Currency: {report['currency'] or 'not declared'} · Date parsing: {report['date_parsing']}")
        if report["rejection_reasons"]:
            st.markdown("**Why rows were rejected**")
            st.dataframe(pd.Series(report["rejection_reasons"], name="rows").rename_axis("reason").reset_index(),
                         hide_index=True, width="stretch")
            st.caption("Examples (row numbers as in the file, header = row 1): " + "; ".join(
                f"row {x['row']}: {', '.join(x['reasons'])}" for x in report["rejected_examples"][:10]))
        if report["missing_value_pct"]:
            st.markdown("**Missing values in mapped columns (%)**")
            st.dataframe(pd.Series(report["missing_value_pct"], name="% missing").rename_axis("field").reset_index(),
                         hide_index=True, width="stretch")
    with right:
        fr = report["forecast_readiness"]
        st.markdown("**Forecasting readiness**")
        st.write(f"Level: **{fr['level']}** · median {fr['median_months']:g} months per series (min {fr['min_months']}) · "
                 f"missing months {fr['missing_month_share_pct'] if fr['missing_month_share_pct'] is not None else '–'}%")
        for n in fr["notes"]:
            st.caption(n)
        st.markdown("**Segmentation readiness**")
        st.write(f"{report['segmentation_readiness']['status']}: {report['segmentation_readiness']['reason']}")
        st.markdown("**Analyses this data supports**")
        caps = report["capabilities"]
        for k, label in (("revenue_analysis", "Revenue and price analysis"), ("marketing_analysis", "Marketing analysis"),
                         ("competitor_analysis", "Competitor analysis")):
            st.write(("✅ " if caps[k]["enabled"] else "⛔ ") + label + ("" if caps[k]["enabled"] else f" ({caps[k]['reason']})"))
        mf = report["model_features"]
        st.caption(f"Model features available: {', '.join(mf['available']) or 'none'}"
                   + (f" · too sparse: {', '.join(mf['too_sparse'])}" if mf["too_sparse"] else ""))
    if monthly_preview:
        st.markdown("**Monthly data that would be ingested (first rows)**")
        st.dataframe(pd.DataFrame(monthly_preview), hide_index=True, width="stretch")


MODEL_LABELS = {"xgboost": "XGBoost", "naive_last_month": "Naive (last month)", "seasonal_naive": "Seasonal naive",
                "xgboost_on_seasonal_rows": "XGBoost (rows with a year of history)"}


def show_version(v: dict) -> None:
    res = v["results"]
    fc = res["forecasting"]
    ev = fc["evaluation"]
    st.markdown(f"**Model version** `{v['version_id'][:8]}` · trained {v['created_at'][:16].replace('T', ' ')} UTC · "
                f"recommended: **{MODEL_LABELS.get(fc['recommended_model'], fc['recommended_model'])}**")
    st.caption(f"Holdout: one-month-ahead predictions for the last {ev['holdout_months']} months "
               f"({ev['test_start']} onward), models fitted only on data up to {ev['train_end']}.")
    rows = [{"model": MODEL_LABELS.get(m, m), **s} for m, s in ev["metrics"].items()]
    if res["prophet"]["status"] == "ok":
        rows.append({"model": f"Prophet ({len(res['prophet']['series'])} longest series, multi-month)",
                     **res["prophet"]["metrics"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("MAE and RMSE are in units sold per region/product/month. WAPE = total absolute error ÷ total units.")
    for n in fc["notes"]:
        st.info(n)
    if res["prophet"]["status"] == "ok":
        st.caption(f"Prophet: {res['prophet']['evaluation']}; not directly comparable with the one-month-ahead rows.")
    else:
        st.caption(f"Prophet: {res['prophet']['reason']}")

    t1, t2, t3 = st.tabs(["Forecasts", "Feature influence (SHAP)", "Segments"])
    with t1:
        f = pd.DataFrame(fc["forecasts"])
        f = f[f["model"] == fc["recommended_model"]]
        st.dataframe(f.pivot_table(index=["region", "product"], columns="target_month", values="y_pred").round(1),
                     width="stretch")
        st.caption(fc["forecast_note"])
    with t2:
        shap = res["explainability"]
        if shap["status"] == "ok":
            imp = pd.DataFrame(shap["global_importance"])
            st.plotly_chart(px.bar(imp.sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature", orientation="h",
                                   labels={"mean_abs_shap": "Mean |SHAP| (units)", "feature": ""},
                                   title=f"Global importance ({shap['method']})"), width="stretch")
        else:
            st.write(shap["reason"])
        st.warning(shap["caveat"])
    with t3:
        seg = res["segmentation"]
        if seg["status"] != "ok":
            st.write(seg["reason"])
        else:
            chosen = next(t for t in seg["k_selection"] if t["chosen"])
            st.write(f"K-Means with k = {seg['k']} (silhouette {chosen['silhouette']:.2f}) on "
                     f"{', '.join(seg['features_used'])}.")
            if seg["features_omitted"]:
                st.caption(f"Not used (missing for some cells): {', '.join(seg['features_omitted'])}")
            st.dataframe(pd.DataFrame(seg["profiles"]), hide_index=True, width="stretch")
            with st.expander("Region/product cells and their segment"):
                st.dataframe(pd.DataFrame(seg["cells"]), hide_index=True, width="stretch")
            st.caption(seg["note"])


def show_training(dataset_id: str) -> None:
    st.subheader("Models for this dataset")
    jobs = call(api_client.training_jobs, key, dataset_id, spinner="Loading training status...")["items"]
    versions = call(api_client.model_versions, key, dataset_id, spinner="Loading model versions...")["items"]
    active = next((j for j in jobs if j["status"] in ("queued", "running")), None)
    if active:
        st.info(f"Training is **{active['status']}** (requested {active['requested_at'][:16].replace('T', ' ')} UTC). "
                "It runs in a separate worker process, usually within a few minutes once the worker is running.")
        if st.button("Refresh status"):
            st.rerun()
    else:
        last = jobs[0] if jobs else None
        if last and last["status"] == "failed":
            st.error(f"The last training attempt failed: {last['error']}")
        if st.button("Train models on this dataset", type="primary"):
            try:
                with st.spinner("Queueing..."):
                    res = api_client.train_models(key, dataset_id)
                st.success(res["message"])
                st.rerun()
            except api_client.ApiError as exc:
                st.warning(exc.message)
        st.caption("Fits XGBoost (with a naive and seasonal-naive baseline), optional Prophet, and K-Means segments "
                   "on this dataset only. Training the same data twice is refused because it would give the same result.")
    if not versions:
        if not active:
            st.caption("No models trained yet.")
        return
    picked = st.selectbox("Model version", [v["version_id"] for v in versions],
                          format_func=lambda i: next(f"{v['created_at'][:16].replace('T', ' ')} · {v['recommended_model']}"
                                                     for v in versions if v["version_id"] == i))
    show_version(call(api_client.model_version, key, picked, spinner="Loading results..."))


# ---------------------------------------------------------------- ingested / validated dataset
if meta["status"] == "ready":
    detail = call(api_client.dataset_detail, key, selected, spinner="Loading the quality report...")
    st.success(f"Ingested {meta['row_count']:,} monthly rows covering {meta['date_start']} to {meta['date_end']}.")
    st.session_state["dataset_id"] = selected
    st.caption("The forecasting, segmentation and optimizer pages still show the synthetic demo; results for this "
               "dataset appear below.")
    with st.expander("Quality report", expanded=False):
        show_report(detail["report"])
    show_training(selected)
    st.stop()

# ---------------------------------------------------------------- mapping and validation
profiles = st.session_state.setdefault("byod_profiles", {})
if selected not in profiles:
    profiles[selected] = call(api_client.dataset_profile, key, selected, spinner="Reading the stored file...")["profile"]
prof = profiles[selected]

st.subheader("1 · Preview")
for w in prof["file_warnings"]:
    st.warning(w)
st.dataframe(pd.DataFrame(prof["preview"]), hide_index=True, width="stretch")
with st.expander("Column profile"):
    st.dataframe(pd.DataFrame(prof["columns"]), hide_index=True, width="stretch")

st.subheader("2 · Map your columns")
st.caption("Suggestions are pre-filled from the column names. Check every one. Unmapped optional fields are left "
           "empty and the analyses that need them are switched off. Nothing is invented.")
columns = [c["name"] for c in prof["columns"]]
options = [NOT_MAPPED, *columns]
stored = meta.get("schema_mapping") or {}
suggested = stored.get("columns") or prof["suggested_mapping"]
if stored and f"byod_body_{selected}" not in st.session_state:
    st.session_state[f"byod_body_{selected}"] = {"mapping": stored["columns"],
                                                 **{k: v for k, v in stored.items() if k != "columns"}}
mapping: dict[str, str] = {}
grid = st.columns(3)
for i, f in enumerate(cfg["fields"]):
    default = suggested.get(f["name"])
    choice = grid[i % 3].selectbox(f"{f['label']}{' *' if f['required'] else ''}", options,
                                   index=options.index(default) if default in options else 0,
                                   key=f"map_{selected}_{f['name']}", help=f["description"])
    if choice != NOT_MAPPED:
        mapping[f["name"]] = choice

with st.expander("Parsing options", expanded="product" not in mapping):
    o1, o2 = st.columns(2)
    dayfirst = o1.checkbox("Dates are day-first (DD/MM/YYYY)",
                           value=bool(stored.get("dayfirst", prof["suggested_options"].get("dayfirst", False))),
                           key=f"dayfirst_{selected}")
    date_format = o1.text_input("Exact date format (optional)", value=stored.get("date_format") or "",
                                placeholder="%d-%m-%Y", key=f"fmt_{selected}")
    currency = o2.text_input("Currency (3-letter code, optional)", value=stored.get("currency") or "", max_chars=3,
                             placeholder="INR", key=f"cur_{selected}")
    drop_dupes = o2.checkbox("Drop rows that exactly duplicate an earlier row",
                             value=bool(stored.get("drop_exact_duplicates", False)), key=f"dupes_{selected}")
    label = None
    if "product" not in mapping:
        label = st.text_input("Single-product label (required when no product column is mapped)", max_chars=80,
                              value=stored.get("single_product_label") or "", placeholder="All products",
                              key=f"label_{selected}")

body = {"mapping": mapping, "dayfirst": dayfirst, "date_format": date_format or None,
        "currency": currency.upper() or None, "drop_exact_duplicates": drop_dupes, "single_product_label": label or None}

st.subheader("3 · Validate")
if st.button("Validate mapping", type="primary"):
    st.session_state[f"byod_report_{selected}"] = call(api_client.validate_dataset, key, selected, body,
                                                       spinner="Validating every row...")
    st.session_state[f"byod_body_{selected}"] = body
    st.rerun()

result = st.session_state.get(f"byod_report_{selected}")
if result is None and meta["status"] in ("validated", "invalid"):
    detail = call(api_client.dataset_detail, key, selected, spinner="Loading the last report...")
    if detail["report"]:
        result = {"dataset": detail["dataset"], "report": detail["report"], "monthly_preview": []}
        st.caption("Showing the report from the last validation. Validate again after changing the mapping.")
if result is None:
    st.stop()
show_report(result["report"], result.get("monthly_preview"))

st.subheader("4 · Confirm ingestion")
if result["dataset"]["status"] != "validated":
    st.info("Resolve the blocking errors above, then validate again.")
    st.stop()
if st.session_state.get(f"byod_body_{selected}", body) != body:
    st.warning("The mapping or options changed since this report was produced. Validate again before ingesting.")
    st.stop()
st.write("Ingestion stores the monthly rows shown above for this dataset only. The uploaded file and the synthetic "
         "demo stay unchanged.")
if st.button("Confirm and ingest", type="primary"):
    call(api_client.activate_dataset, key, selected, spinner="Ingesting...")
    st.session_state.pop(f"byod_report_{selected}", None)
    st.rerun()
