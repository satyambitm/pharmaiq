# AI Business Analyst

Page 7 and `POST /api/v1/agent/chat` answer plain-language questions with a LangChain tool-calling loop on
OpenAI (`gpt-4o-mini` by default). The model can only call six read-only tools; it never sees SQL, files, code
execution or other users' data.

## How a question is answered

1. The server picks the dataset **before** the model runs: the synthetic demo, or an uploaded dataset that the
   caller's API key owns (`dataset_id` in the request, checked like every dataset endpoint). No tool takes a
   dataset argument, so the model cannot switch datasets.
2. The model calls tools (at most `AGENT_MAX_TOOL_CALLS`, default 6) within `AGENT_TIMEOUT_SECONDS`. Bad
   arguments come back to it as error data so it can correct itself.
3. The server returns the answer with the tools used and their arguments, the caveats those tools attached (SHAP
   is association, budget results are simulations), and a **grounding check**: every number in the answer is
   compared with the numbers in the tool results and the user's messages. Anything unmatched is listed under
   `grounding.unverified` and shown as a warning on page 7. Small integers (counts, months) are not checked.

| Tool | Demo | Uploaded dataset |
| --- | --- | --- |
| `get_sales_summary` | KPIs, regions, product mix | Units; revenue and marketing only when the data supports them |
| `get_sales_forecast` | Live XGBoost or stored Prophet, with 2025 backtest accuracy | Latest trained version's forecasts and holdout metrics |
| `get_market_segments` | Stored K-Means segments | Latest version's segments, or why they were skipped |
| `explain_forecast` | Stored SHAP importance + caveat | Latest version's SHAP + caveat |
| `compare_budget_scenarios` | Optimizer under assumed curves + label | Refused: no defensible response model |
| `get_data_quality` | Synthetic data facts | The ingestion quality report |

The system prompt requires numbers to come from tools, forbids causal or ROI claims, treats tool output as data
(not instructions), and tells the model to state limitations instead of working around them.

## Configuration

| Variable | Default | Notes |
| --- | --- | --- |
| `AGENT_MODE` | `disabled` | `public`: anyone may ask about the demo. `api_key`: every chat needs `X-API-Key` (same `API_KEYS`) |
| `OPENAI_API_KEY` | – | Secret. Local `.env` (gitignored) or the Render dashboard only. Never sent to the frontend |
| `AGENT_MODEL` | `gpt-4o-mini` | Any OpenAI chat model with tool calling |
| `AGENT_MAX_TOOL_CALLS` | `6` | 1–12 per question |
| `AGENT_TIMEOUT_SECONDS` | `30` | Per question |
| `AGENT_REQUESTS_PER_HOUR` | `20` | Per client address (or per owner for uploaded data) |
| `AGENT_REQUESTS_PER_DAY` | `300` | Whole service; bounds spend |

The agent is on only when `AGENT_MODE` is not `disabled` **and** a key is set. The limits are in-process: they
reset on restart and are per instance, which suits the single free instance; they bound cost, not identity. On
Render the client address is the last `X-Forwarded-For` entry (added by Render's proxy).

Logs record the tools used, the number of unverified figures and the duration, never the question text, the
answer or provider error messages (they can echo request details).

## Running locally

```powershell
# .env: OPENAI_API_KEY=sk-...
$env:DATA_BACKEND="csv"; $env:AGENT_MODE="public"
uvicorn backend.app.main:app --port 8000
cd frontend; streamlit run app.py      # open "AI Business Analyst"
```

Tests (`backend/tests/test_agent_api.py`) use a scripted fake model and never call OpenAI.

## Deployment and rollback

`render.yaml` sets `AGENT_MODE=public` with 10 questions per hour per visitor and 200 per day, and declares
`OPENAI_API_KEY` with `sync: false` so its value lives only in the dashboard. Without the key the endpoint answers
503 `agent_disabled` and page 7 says the analyst is off.

Rollback: set `AGENT_MODE=disabled` (or delete `OPENAI_API_KEY`) in the Render dashboard; the next restart turns
the agent off with no code change. If the key may have leaked, revoke it in the OpenAI dashboard first. No
database changes are involved.
