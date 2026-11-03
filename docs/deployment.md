# Deployment guide: Render (FastAPI) + Streamlit Community Cloud

One GitHub repository, two free services:

```text
Browser ──► Streamlit Community Cloud (frontend/app.py)
                │  HTTPS, API_BASE_URL from Streamlit Secrets
                ▼
            Render web service (uvicorn backend.app.main:app)
                │  loads models/ once at startup (committed, ~0.9 MB)
                ▼
            Render Postgres (free)  ── or ──  bundled CSV (DATA_BACKEND=csv)
```

Hosting facts below were checked against the official docs in November 2026. Free tiers change, so recheck before relying on them.

## Free-tier realities

| Platform | Limit | Effect on this project |
|---|---|---|
| Render free web service | 0.1 CPU, 512 MB RAM; spins down after 15 minutes without traffic; waking takes about a minute; local filesystem is not persistent | First request after idle is slow. The frontend uses a 90 s read timeout, retries 502/503/504 twice and tells the user the API may be waking. Measured API memory: about 200 MB resident after serving every endpoint. Nothing is written to disk at runtime. |
| Render free Postgres | 1 GB; **expires 30 days after creation** (14-day grace period to upgrade); no backups | The API reseeds an empty database automatically (`AUTO_SEED_DEMO=true`), so a recreated database needs no manual load. When it expires, switch to the CSV fallback below. |
| Render pre-deploy command | Paid instances only | That is why seeding runs at API startup instead. |
| Render default Python | 3.14 for new services | Pinned to 3.11.9 with `PYTHON_VERSION` (and `.python-version`). |
| Streamlit Community Cloud | About 0.08–2 CPU, 0.7–2.7 GB RAM; apps hibernate after 12 hours without traffic | A visitor clicks "Yes, get this app back up!" to wake it. The frontend needs only `streamlit`, `requests`, `pandas` and `plotly`. |

Prophet and SHAP are never installed on Render: their results are computed offline and shipped in `models/`. XGBoost inference runs live (under 1 s per region).

## 1. Push the project to GitHub

```bash
git remote add origin https://github.com/<your-user>/pharmaiq.git
git push -u origin main
```

Check that `.env` and any `secrets.toml` are **not** in the repository (`git ls-files | grep -E "\.env$|secrets.toml$"` should print nothing). `models/` and `data/sample/` **must** be committed; the API cannot start without them.

## 2. Deploy FastAPI to Render

**Option A, Blueprint (recommended).** In the Render Dashboard choose **New → Blueprint**, connect the repository and confirm. `render.yaml` creates:

- `pharmaiq-api`: free Python web service in Singapore.
  - Build: `pip install -r backend/requirements.txt`
  - Start: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
  - Health check: `/health/ready` (fails until the data store and model bundle are both available)
- `pharmaiq-db`: free Postgres in the same region, with external connections blocked (`ipAllowList: []`). The service reaches it over Render's private network.

**Option B, manual web service.** New → Web Service → the repo, Language Python 3, Root Directory blank (repo root), the same build/start commands and health check path, Instance Type Free.

### Environment variables

| Key | Value | Notes |
|---|---|---|
| `PYTHON_VERSION` | `3.11.9` | Required; Render's default is newer than the pinned wheels target |
| `DATA_BACKEND` | `postgres` or `csv` | `csv` needs no database |
| `DATABASE_URL` | from the database (`connectionString`) | Blueprint wires it; manually, paste the **Internal** Database URL. Never commit it. `postgres://` and `postgresql://` prefixes are normalised to the psycopg 3 driver. |
| `AUTO_SEED_DEMO` | `true` | Loads `data/sample/synthetic_sales_clean.csv` if `monthly_sales` is missing or empty; otherwise does nothing |
| `ENVIRONMENT` | `production` | |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` | `1` | Keeps native thread pools (and memory) small on 0.1 CPU |

The database password lives only in Render's environment. Logs never print it: settings store the URL as a secret value, and database errors are logged by exception type only.

## 3. Verify the backend

Replace `<api>` with the URL Render shows on the service page (`https://<name>.onrender.com`). The first call after idle can take about a minute.

```bash
curl https://<api>/health                       # {"status":"ok","version":"1.1.0"}
curl https://<api>/health/ready                 # status "ready", data_rows 1080, data_window 2021-01..2025-12, model versions
curl "https://<api>/api/v1/forecasts?region_id=R01&horizon=3&model=xgboost"
curl -X POST https://<api>/api/v1/optimize-budget -H "Content-Type: application/json" -d '{"total_budget_cr": 100}'
```

Then open `https://<api>/docs` for the interactive OpenAPI page. In the Render logs, look for `startup ... models_loaded=True` and, on the first boot with an empty database, `demo_seed_loaded rows=1080 reconciled=True`.

If `/health/ready` returns 503, its `problems` list says why (database unreachable, or the model bundle failed to load because of a version mismatch).

## 4. Deploy Streamlit to Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub and click **Create app → Deploy a public app from GitHub**.
2. Repository: your repo. Branch: `main`. Main file path: `frontend/app.py`.
3. **Advanced settings:** Python version **3.11**. Secrets:

   ```toml
   API_BASE_URL = "https://<api>.onrender.com"
   ```

   Use the HTTPS Render URL, never `localhost`.
4. Deploy. Community Cloud installs `frontend/requirements.txt` (the file next to the entrypoint takes precedence over the root `requirements.txt`).

To change the API URL later: App → Settings → Secrets, edit, save. The app restarts.

After a push, Community Cloud pulls the new code but can keep already-imported modules such as
`frontend/services/ui.py` in memory, which shows up as errors like `unexpected keyword argument`. Fix: **Manage app →
⋮ → Reboot app**. Render may also not auto-deploy a push; use **Manual Deploy → Deploy latest commit** and check
`/health/ready` and the version in `/docs` afterwards.

## 5. Test the deployed application

- The sidebar shows `API: https://<api>...` and **Backend ready · postgres · 2021-01..2025-12**.
- Overview: net revenue ₹484.1 cr over 60 months, five charts.
- Sales Forecasting: switch region, model and horizon. XGBoost should show `live`, Prophet `precomputed`; the Prophet coverage warning should appear.
- Model Insights: horizon scorecard, coverage table, SHAP charts.
- Budget Optimizer: ₹100 cr base case gives ₹237.69 cr assumed gain, ₹+4.30 cr vs equal split. A ₹10 cr budget shows the feasible-range error.
- Cold start: leave both apps idle, reload. The sidebar may show "not ready" for up to a minute; **Retry connection** recovers once Render is awake.

## 6. Add live links

Only after the checks above pass, add the real URLs to the top of `README.md`:

```markdown
**Live demo:** [Dashboard](https://<app>.streamlit.app) · [API docs](https://<api>.onrender.com/docs)
(free tiers: the API sleeps after 15 min idle and takes about a minute to wake)
```

On a resume: `PharmaIQ — FastAPI + Streamlit forecasting platform (live demo: <dashboard URL>)`.

## Fallbacks

- **Postgres expired or unavailable:** set `DATA_BACKEND=csv` on the Render service and redeploy. The API serves the same 1,080 rows from the committed CSV (tests check that both backends give identical results). `DATABASE_URL` is then not needed.
- **Another managed Postgres** (for example Neon or Supabase free tiers; check their current limits): set `DATABASE_URL` to its connection string, including `?sslmode=require` if the provider requires TLS, and keep `AUTO_SEED_DEMO=true`, or load it yourself with `python scripts/bootstrap_demo.py --load-db` or `psql "$DATABASE_URL" -f sql/schema.sql -f sql/seed.sql`.
- **Render unavailable:** the API is a standard ASGI app. Any host that can run `pip install -r backend/requirements.txt` and `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT` with Python 3.11 works; set the frontend's `API_BASE_URL` to its HTTPS URL.
- **Model artifacts too large for Git** (not the case now, ~0.9 MB): publish the bundle as a GitHub Release asset and download it in the build command, verifying `models/manifest.json` hashes with `scripts/bootstrap_demo.py`.

## Not verified here

This guide has been tested locally (both servers together, Postgres and CSV backends, empty-database auto-seed, `sql/seed.sql`). Linux dependency resolution for Python 3.11 was checked with `pip install --dry-run --platform manylinux...` (35 binary wheels, `xgboost-cpu`, no GPU packages). An actual Render or Streamlit Cloud deployment has **not** been performed yet, and no public URLs exist until you create them.
