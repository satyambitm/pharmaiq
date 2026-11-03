# Progress

## Status: v1.0.0 — all must-have features complete

| Area | Status |
|---|---|
| Seeded synthetic generator + metadata + dirty copy | Done |
| Validation, repair rules, audit report | Done |
| PostgreSQL schema, constraints, idempotent loader, reconciliation | Done |
| 10 SQL analyses + 3 views, reconciled with pandas | Done |
| EDA: 9 figures, 5 documented findings | Done |
| Leakage-safe features + leakage tests | Done |
| Seasonal naive, last-month naive, XGBoost, Prophet on identical rows; 3 folds + 2025 test | Done |
| SHAP global + local | Done |
| K-Means (train-period only), silhouette, seed stability | Done |
| Constrained budget optimizer, scenarios, sensitivity, infeasibility handling | Done |
| Streamlit overview + 5 pages | Done |
| Tests: 25+ automated (DB tests skip if Postgres is down) | Done |
| README, model card, assumptions, data dictionary, business case, interview Q&A | Done |
| Stretch: Docker Compose for Postgres | Done |

## Remaining / future work
- Prediction intervals from backtest residual quantiles, with coverage checks.
- Recursive or direct 3-month-ahead forecasts, evaluated separately.
- Optional 30-territory grain for more robust segmentation (would require a schema and grain change from day one).
- CI workflow running `pytest` with a Postgres service.
- Dashboard screenshots/GIF in the README (current README uses the generated figures).
