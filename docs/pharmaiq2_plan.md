# PharmaIQ 2.0: assessment and plan

PharmaIQ 2.0 extends the existing repository. Nothing is rebuilt; the synthetic demo stays exactly as deployed.

## 1. Current architecture

- **Frontend:** Streamlit (`frontend/`). Five pages call the API through `services/api_client.py` (GET-only
  retries, `API_BASE_URL` from env or secrets). Deployed on Streamlit Community Cloud.
- **Backend:** FastAPI (`backend/app`) on Render's free web service in Singapore. The lifespan builds a
  `SalesRepository` (Postgres `monthly_sales` filtered to `synthetic_v1`, or a CSV fallback), optionally seeds the
  demo, and loads the `ModelBundle` from `models/`. It has request-ID logging, uniform JSON errors and GZip.
- **Models:** recursive XGBoost (live, horizon ≤ 3), precomputed Prophet forecasts, K-Means over 18
  region×product cells, precomputed SHAP JSON, and a scenario optimizer with assumed response curves.
  Training is offline (`ml/train_*.py`); artifacts are committed.
- **Data:** Postgres `pharmaiq-db` on the free plan (expires about 2026-12-03 unless upgraded). Render's disk is
  ephemeral.

## 2. Reused as-is

`SalesRepository` and all demo endpoints; error handling (`AppError` and its subclasses); request-ID
middleware; settings via `pydantic-settings`; `src/features/build.py` (lags and rolling windows) and
`src/modeling/recursive.py` for per-dataset XGBoost; `src/segmentation/kmeans.py`; the optimizer; Streamlit's
`ui.page` and `ui.call` helpers; the test fixtures.

## 3. Missing

Authentication; per-owner storage; safe file parsing; column mapping and data-quality reporting; a persistent
training job queue and model registry; per-dataset model loading; the agent and its tools; the two new pages.

## 4. File-by-file plan

| Milestone | Files |
| --- | --- |
| 2 Ingestion (done) | `src/ingest/{fields,files,pipeline}.py`, `backend/app/core/{auth,config,errors}.py`, `backend/app/db/datasets.py`, `backend/app/schemas/datasets.py`, `backend/app/services/dataset_service.py`, `backend/app/api/routes/datasets.py`, `backend/app/main.py`, `backend/app/api/deps.py`, `frontend/pages/6_Data_Management.py`, `frontend/services/{api_client,ui}.py`, `scripts/make_api_key.py`, `scripts/make_byod_samples.py`, tests, `docs/byod.md` |
| 3 Training | `training_jobs` and `model_versions` tables (Postgres, artifacts in `bytea`); `ml/train_dataset.py` (XGBoost + K-Means + SHAP per dataset, using the existing feature and model code); `scripts/run_training_worker.py` (claims jobs with `SELECT … FOR UPDATE SKIP LOCKED`); routes `POST /api/v1/models/train`, `GET /jobs/{id}`, `GET /versions`; a training section on page 6 |
| 4 Agent | `backend/app/agent/{tools,runner}.py` (LangChain tool calling; six read-only tools bound server-side to the authenticated owner and the selected dataset); `POST /api/v1/agent/chat` |
| 5 UI | `frontend/pages/7_AI_Business_Analyst.py` |
| 6 Deploy | `render.yaml` environment additions (secrets set in the dashboard); live verification |

## 5. Dependencies and environment variables

- Added in Milestone 2: `python-multipart==0.0.33`, `openpyxl==3.1.5`. Both are small, pure-Python packages.
- Variables added in Milestone 2: `UPLOADS_MODE`, `API_KEYS`, `DATASET_STORE_URL`, `MAX_UPLOAD_MB`,
  `MAX_UPLOAD_ROWS`, `MAX_DATASETS_PER_OWNER` (see `docs/byod.md`).
- Planned for Milestone 4: `langchain-core` plus one provider package; `LLM_PROVIDER`, `LLM_MODEL`,
  `LLM_API_KEY` (a Render secret only), `AGENT_MAX_TOOL_CALLS`, `AGENT_DAILY_LIMIT`.
- Prophet is not installed on Render and will not be. Per-dataset Prophet training runs only in the worker
  environment, which has the root `requirements.txt`.

## 6. Database and storage changes

Milestone 2 adds `datasets`, `dataset_files`, `dataset_rows` and `schema_migrations`. They are created only when
uploads are enabled, and the demo tables are untouched. Milestone 3 will add `training_jobs` and
`model_versions`. Everything is additive; rollback is in `docs/byod.md`.

## 7. Security before public uploads

- Real authentication (`UPLOADS_MODE=api_key`, hashed keys, constant-time comparison). The `local` mode is
  refused in production, and the public deployment stays `disabled` until keys are issued.
- Owner scoping in every query; 404 for other owners' data; UUID path parameters.
- Content-based file checks: magic bytes, macro, ZIP-bomb and OLE rejection; size, row, column and
  dataset-count limits; body size checked before parsing.
- No raw keys, database URLs or file contents in logs; error messages are safe to show.
- Agent (Milestone 4): the dataset id comes from the server session, never from the model; tools are read-only
  with no SQL, shell, file or Python access; per-owner rate limits.

## 8. Render and Streamlit considerations

- Free web service: 512 MB RAM, sleeps when idle, ephemeral disk, no background workers. Uploads therefore live
  in Postgres. Training never runs at startup or per request. Milestone 3 uses a Postgres-backed job queue
  processed by a worker command, run locally/offline or on a paid Render worker if one is approved. The API only
  enqueues jobs and serves finished artifacts.
- The free Postgres plan has 1 GB and expires; upgrade it or export the data before 2026-12-03.
- Streamlit Cloud keeps only `API_BASE_URL` in its secrets. Access keys are typed by the user into a password
  field and kept only in that browser session.

## 9. Implementation order

1. ✅ Inspect and plan (this document).
2. ✅ Ingestion: upload, mapping, quality report, confirm, isolation, tests, page 6.
3. Training jobs and model versions; then the training controls on page 6.
4. Agent backend with the six tools and grounded, cited answers.
5. Page 7 (AI Business Analyst).
6. Deploy with uploads still disabled, verify live, then enable `api_key` mode with issued keys.
