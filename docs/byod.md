# Bring your own data (uploads)

PharmaIQ can ingest a user's own sales file next to the synthetic demo. Uploaded data is stored in separate
tables, keyed by owner and dataset, so the demo (`monthly_sales`, `data_source='synthetic_v1'`) is never read or
written by the upload code. All other pages still run on the synthetic demo. Training models on uploaded data
comes in a later milestone and is not available yet.

## Workflow

1. **Upload** (`POST /api/v1/datasets/upload`, multipart `file` + optional `dataset_name`). The file is
   checked (see *File safety*), stored unchanged, and profiled: preview, column types and suggested mapping.
2. **Map and validate** (`POST /api/v1/datasets/{id}/validate`). The user maps their columns to PharmaIQ fields
   and chooses parsing options. The API cleans and aggregates the file *in memory* and returns a quality report
   plus a preview of the monthly rows. Nothing is ingested at this step.
3. **Confirm** (`POST /api/v1/datasets/{id}/activate`). This re-reads the stored file, re-applies the stored
   mapping (deterministic, so the result matches the report) and writes the monthly rows in one transaction.
4. **Manage**: `GET /api/v1/datasets`, `GET /{id}` (metadata and latest report), `GET /{id}/profile`,
   `DELETE /{id}` (removes the file, report and rows).

`GET /api/v1/datasets/settings` is public: it tells the UI whether uploads are enabled, what the limits are and
which fields can be mapped.

Statuses: `uploaded` → `invalid` | `validated` → `ready`. An ingested (`ready`) dataset cannot be re-mapped;
upload the file again instead. This keeps each version's report and data consistent.

## Fields

Required: **date**, **region**, **units sold**. **Product** is also required unless the user gives an explicit
single-product label. Optional: unit price, revenue, marketing spend, doctor interactions, competitor price,
competitor count, market size, HCP count, currency, unit of measure.

## Data rules (nothing is fabricated)

- Rows with a missing or invalid date, region, product or units, negative numbers, zero price, dates before 1990
  or in the future are **rejected and counted**, with example row numbers. Rejected rows are never "fixed".
  More than 50 % rejected, or none valid, blocks ingestion.
- Numbers: `1,234.5`, currency symbols and codes (`₹`, `Rs`, `INR`, `$`, `USD`, `€`, `£`) are accepted.
  Ambiguous European formats like `1.234,5` are rejected rather than guessed.
- Dates: ISO, `YYYY-MM`, `Jan 2024`, and numeric `DD/MM/YYYY` or `MM/DD/YYYY`. Day-first is suggested from the
  data. When nothing in the file settles it, a warning says how many dates are ambiguous.
- Daily or weekly rows are aggregated to months. Units, revenue, marketing spend and doctor interactions are
  summed. Unit price is revenue ÷ units, so it is weighted by units. Competitor and market fields are monthly
  means. A monthly sum is left **empty** if any contributing row lacked the value.
- **Price missing:** revenue and price analysis is disabled. Prices are never invented. **Marketing missing:**
  marketing analysis is disabled and the report says why.
- Mixed currencies or mixed units of measure block ingestion. Missing months are reported, never filled.
- Exact duplicate rows are reported. They are dropped only if the user ticks the option.
- The report also gives forecasting readiness (24+ months = full; 12–23 = short history, no yearly seasonality;
  6–11 = baseline only; under 6 = insufficient) and segmentation readiness (at least 3 regions and 6 cells).

## File safety

- Only `.csv` and `.xlsx` files are accepted. The **content** is checked, not just the name: binary data renamed
  to `.csv`, CSV renamed to `.xlsx`, legacy `.xls`/OLE files, ZIPs that aren't workbooks, macro-enabled workbooks
  (`vbaProject.bin`), and ZIP bombs (entry count, total expanded size and compression ratio) are all rejected.
- Limits (configurable): 5 MB per file, 200,000 rows, 10 datasets per owner, 100 columns. Request bodies over
  the limit are refused with 413 *before* parsing.
- Headers must be present and unique, and rows must have a consistent field count. Only the first sheet of a
  workbook is read, with a warning if there are more.
- Cell values are only ever treated as data. They are not evaluated as formulas, SQL or code.

## Authentication and isolation

`UPLOADS_MODE` controls access:

| Mode | Use | Behaviour |
| --- | --- | --- |
| `disabled` (default) | Public demo | Upload endpoints return 403; the page explains uploads are off |
| `local` | Your own machine | No credentials; single owner `local-dev`. **Refused at startup when `ENVIRONMENT=production`** |
| `api_key` | Shared or private deployment | Every request needs `X-API-Key`. The server stores only SHA-256 hashes in `API_KEYS` |

The owner always comes from the authenticated key, never from the request. Every query filters by owner.
Another owner's dataset returns 404, not 403, so its existence is not revealed. Keys are compared in constant
time. Raw keys and database URLs are never logged.

Issue a key:

```bash
python scripts/make_api_key.py alice
# prints the raw key (give it to Alice once) and an entry like  alice:3f2a...  to append to API_KEYS
```

## Environment variables

| Variable | Default | Notes |
| --- | --- | --- |
| `UPLOADS_MODE` | `disabled` | `disabled`, `local` or `api_key` |
| `API_KEYS` | – | `owner:sha256hex,owner2:sha256hex`. Secret; set it in the Render dashboard, never in git |
| `DATASET_STORE_URL` | `DATABASE_URL` | Where uploads are stored. SQLite works locally (`sqlite:///./byod_dev.db`) |
| `MAX_UPLOAD_MB` | `5` | 0–50 |
| `MAX_UPLOAD_ROWS` | `200000` | 10–1,000,000 |
| `MAX_DATASETS_PER_OWNER` | `10` | 1–100 |

## Running locally

```powershell
$env:DATA_BACKEND="csv"; $env:UPLOADS_MODE="local"; $env:DATASET_STORE_URL="sqlite:///./byod_dev.db"
uvicorn backend.app.main:app --port 8000
# in another shell
cd frontend; streamlit run app.py      # open "Data Management"
```

Sample files in `data/byod_examples/` (synthetic, made by `scripts/make_byod_samples.py`) cover a clean monthly
file with deliberate defects, a daily pharmacy file with day-first dates and no price, and a single-product
units-only file.

## Storage

Four tables are created on startup only when uploads are enabled, using `CREATE TABLE IF NOT EXISTS` semantics.
They are recorded in `schema_migrations` as `001_datasets`:

- `datasets`: one row per upload (owner, status, file hash, mapping, quality report)
- `dataset_files`: the original bytes. Render's disk is ephemeral, so files live in Postgres
- `dataset_rows`: ingested monthly rows, primary key `(dataset_id, month_start, region, product)`
- `schema_migrations`

The demo tables are untouched. Deleting a dataset deletes its rows in all three tables.

## Deployment and rollback

The public Render deployment keeps `UPLOADS_MODE` unset, which means `disabled`, so this release changes nothing
for public visitors except that the new page explains uploads are off. To enable private uploads on Render, set
`UPLOADS_MODE=api_key` and `API_KEYS` as secret environment variables. The free Postgres plan has 1 GB; at the
5 MB × 10 datasets limit, one owner uses at most about 50 MB of raw files.

Rollback:

1. Set `UPLOADS_MODE=disabled` (or unset it) and redeploy. Upload endpoints stop immediately; the demo is
   unaffected.
2. To remove stored uploads, run against the database:
   `DROP TABLE IF EXISTS dataset_rows, dataset_files, datasets, schema_migrations;`
3. To roll back the code, revert the release commit. The demo tables and endpoints are unchanged by this release.
