# Assumptions

Everything in this project rests on the assumptions below. All data is **synthetic**.

## 1. Data generation (`gen-eq-1.0`)

For series (region r, product p) at month index t (0 = Jan 2021):

```
units = base_p × region_mult_r
        × (1 + (trend_p + trend_r) × t)                         linear trend
        × (1 + amp_p × sin(2π (month − phase_p) / 12))          annual seasonality
        × regional_demand_index[r, t]                           AR(1), φ = 0.6, sd = 0.03
        × (1 − 0.25 × (competitor_index[r, p, t] − 1))          AR(1), φ = 0.8, sd = 0.05
        × (1 + 0.12 × (1 − exp(−exposure_t / base_spend)))      capped concave marketing effect (≤ 12%)
        × (1 + 0.05 × campaign_flag)                            mild campaign lift
        × sqrt(active_accounts / expected_accounts)             coverage
        × lognormal(0, σ_p)                                     noise
exposure_t = 0.6 × spend[t−1] + 0.4 × spend[t]
```

| Product | base units | 2021 list price (INR) | monthly trend | seasonal amp / peak phase | noise σ | base discount |
|---|---|---|---|---|---|---|
| Therapy A | 4,000 | 850 | 0.4% | 12% / 1 | 0.05 | 8% |
| Therapy B | 2,500 | 1,400 | 0.8% | 20% / 4 | 0.07 | 12% |
| Therapy C | 1,200 | 2,600 | 1.2% | 8% / 7 | 0.10 | 5% |

Region multipliers: North 1.30, South 1.15, East 1.00, West 1.10, Central 0.75, Northeast 0.55, with small region trend adjustments (−0.1% to +0.4% per month).

- Marketing spend is a *plan*: ~8% of expected revenue × seasonal budget curve × 1.6 in campaign months × lognormal(0, 0.15). It does **not** depend on realized units, so there is no backdoor from units to spend.
- Spend and units are both raised in campaign months, so a naive spend-vs-units correlation is confounded by design.
- Money is computed in integer paise, so reconciliation is exact.

## 2. Available at prediction time

To forecast month T, made at the end of month T − 1:

| Known | Not known (never used as an input) |
|---|---|
| All units, spend, price, competitor index, accounts and campaign flags through T − 1 | Realized units, revenue, spend, price, discount, competitor index in T |
| Calendar month of T | Whether T is a campaign month |

## 3. Forecast evaluation

- Train targets ≤ Dec 2023; validation 2024; **untouched test 2025**.
- XGBoost tuning: 8 configurations, mean WAPE over 3 expanding folds (validation windows Jul–Dec 2023, Jan–Jun 2024, Jul–Dec 2024).
- Final XGBoost refit on targets through Dec 2024, then evaluated once on 2025. Within 2025, inputs at each origin are actual observed history (rolling-origin one-step evaluation); the model itself is not refit.
- Prophet: refit for each series and each origin on history strictly before the target month, default additive yearly seasonality, no regressors.
- Negative predictions are clipped to 0 for every model. In the v1.0 run no XGBoost prediction was negative (minimum raw prediction ≈ 914 units), so clipping changed nothing.
- The Jan 2026 revenue estimate assumes the last observed (Dec 2025) realized price and the trailing 12-month discount rate. It ignores the January price-increase pattern in the generator, so it is likely understated.

## 4. Segmentation

- Unit: 18 region × product market cells; descriptors from Jan 2021 – Dec 2023 only.
- K chosen from 2–4 by silhouette, requiring every cluster to have ≥ 2 cells, and preferring the smaller K when silhouettes are within 0.01. The tie rule was added after seeing that K = 3 (0.448) and K = 4 (0.449) were virtually tied on the first run.

## 5. Budget simulator (`src/optimization/response.py`)

`gain_r(b) = α_r × ln(1 + b / scale_r)`, all in INR crore. Hand-set, **not estimated** from data, SHAP or correlations.

| Region | α (cr) | scale (cr) | min (cr) | max (cr) | assumed status-quo share |
|---|---|---|---|---|---|
| North | 60 | 15 | 5 | 30 | 24% |
| South | 45 | 12 | 5 | 30 | 20% |
| East | 40 | 10 | 5 | 30 | 17% |
| West | 55 | 14 | 5 | 30 | 19% |
| Central | 30 | 8 | 5 | 30 | 12% |
| Northeast | 25 | 6 | 5 | 30 | 8% |

- Default budget ₹100 crore; feasible range with default bounds is ₹30–180 crore.
- Scenarios multiply every α by 0.8 / 1.0 / 1.2. A uniform multiplier does not change the optimal allocation, only total gain.
- Competitor pressure reduces the α of selected regions by a chosen fraction, which does change the allocation.
- The objective is concave and constraints are linear, so the SLSQP optimum is global.
