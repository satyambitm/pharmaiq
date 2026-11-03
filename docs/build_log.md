# Build log

All steps were run on Windows 11, Python 3.11.9, in a single session (compressed from the 14-day plan at the owner's request).

| # | Task | Command / result |
|---|---|---|
| 1 | Install Python 3.11 | `winget install Python.Python.3.11` → 3.11.9 |
| 2 | Repo + venv | `git init`, `py -3.11 -m venv .venv`, pip 26.3 |
| 3 | Dependencies | `pip install -r requirements.txt` (xgboost 3.0.5, prophet 1.1.7, shap 0.48.0, streamlit 1.66.1) |
| 4 | PostgreSQL | `docker compose up -d` → `postgres:16-alpine`, healthy |
| 5 | Generate, validate, load | `python scripts/bootstrap.py` → 1,080 clean rows, 1,083 dirty rows, 26 anomalies; clean passed; DB reconciled exactly |
| 6 | Tests (first pass) | `pytest` → 25 passed |
| 7 | Reports | `python scripts/build_reports.py` → 9 EDA figures, 10 SQL queries, SQL vs pandas checks all true |
| 8 | K tie-break | K = 3 vs K = 4 silhouettes 0.448 vs 0.449; added a predeclared parsimony rule (within 0.01 → smaller K); K = 3 chosen |
| 9 | Training (first attempt) | Failed: `AttributeError: 'Prophet' object has no attribute 'stan_backend'`. Root cause: cmdstanpy 1.3.0 rejects Prophet's bundled CmdStan (no `makefile`). Fix: pin `cmdstanpy>=1.2,<1.3` |
| 10 | Training | `python scripts/train_all.py` → 2025 WAPE: Prophet 7.88, XGBoost 8.06, last-month 9.58, seasonal 9.79; Prophet 540 refits total |
| 11 | SHAP figure fix | Waterfall title overlapped the f(x) label; added title padding and regenerated |
| 12 | Dashboard | `streamlit run dashboard/app.py`; all 6 pages loaded with no exceptions; infeasible budget (₹10 cr) shows the feasible-range error |
