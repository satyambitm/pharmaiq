# Interview Q&A — PharmaIQ Lite+

Numbers quoted are from the v1.0 run on synthetic data.

## 7-minute walkthrough
1. **(30 s) Context and disclaimer.** Fictional Aster Pharma, 3 products × 6 regions × 60 months, fully synthetic with a documented, seeded equation. Nothing here is real market evidence.
2. **(1 min) Data and SQL.** Generator → validation (clean must pass; a dirty copy with 26 logged anomalies is detected and repaired) → PostgreSQL with PK/FK and CHECK constraints, including exact `NUMERIC` reconciliation. Show q10 (calendar cross join → 0 missing keys) and q02 (`LAG` with `NULLIF`).
3. **(1 min) EDA.** 7.6% unit CAGR; top three regions about 61% of revenue; Therapy B seasonal amplitude 44%; campaign comparison is confounded.
4. **(2 min) Forecasting.** One-month-ahead contract, `shift(1)` before `rolling`, leakage test. On the same 216 rows in 2025: Prophet 7.88% WAPE, XGBoost 8.06%, last-month naive 9.58%, seasonal naive 9.79%. XGBoost won every backtest fold. Per region the winner varies.
5. **(1.5 min) SHAP and K-Means.** `lag_12` and `lag_1` dominate SHAP. K-Means picks K = 3 and the clusters are the products. Say why: product parameters dominate.
6. **(1.5 min) Budget simulator.** Concave assumed curves, SLSQP, equal marginal gain at the optimum (1.545 per crore), infeasible-budget warning, competitor-pressure scenario.
7. **(1 min) Limitations and next steps.** Synthetic data; N = 18 for clustering; assumed response curves; next: prediction intervals, real data, geo-experiments.

## Data / SQL
**What is the grain?** One row per month × region × product; 1,080 rows. Every join, aggregate and forecast is defined at that grain.

**Why normalize regions and products?** Names live in one place, foreign keys stop invalid IDs, and renaming a region is a single update.

**Why `NUMERIC(14,2)` for money?** Binary floats can't represent 0.10 exactly, so sums drift. `NUMERIC` is exact, which lets the database enforce `net = gross − discount` with no tolerance.

**How did you detect missing months?** `generate_series` builds the full calendar, a cross join with regions and products gives every expected key, and a left join to the fact table finds the gaps (q10). The same check runs in pandas in `validate.py`.

**Business value of `LAG`?** Month-over-month and year-over-year growth without self-joins. `LAG(x, 12)` is only valid because the panel has been proven complete.

**Why can't CMS Part D replace this dataset?** It is annual, prescriber-level fills and drug cost. It is not a company's monthly recognized net revenue, so it has the wrong grain and the wrong definition for this forecast.

**Why is average-of-averages misleading?** A small region's average counts as much as a large region's. Weighted measures such as WAPE and the weighted discount % weight by volume.

## Forecasting
**Why not a random train/test split?** Rows close in time are correlated, and the model would see the future. Splits are by target month: train ≤ 2023, validation 2024, test 2025.

**What exactly is one-step-ahead?** Standing at the end of month T − 1, predict month T. In 2025 each prediction uses actual history up to the prior month; the model itself is fixed.

**How did you prevent leakage?** Only origin-month values are inputs. Every rolling statistic uses `groupby().shift(1).rolling()`. A test multiplies all units after June 2024 by 100 and asserts that earlier feature rows are unchanged.

**Why a seasonal naive baseline?** It is a strong, free benchmark for annual seasonality. Here it is biased low because the series grow about 7.6% a year, so last-month naive is slightly better.

**What does Prophet model?** An additive piecewise-linear trend with changepoints, plus Fourier-series yearly seasonality, fit with Stan. It takes `ds`/`y` only. I refit it per series at every origin.

**When did XGBoost do better or worse?** It was better on all three 2023–2024 folds (for example 8.26% vs 10.47% WAPE on Jan–Jun 2024). It was slightly worse than Prophet on 2025 (8.06% vs 7.88%). Prophet improved as more seasonal cycles became available.

**How did you make the comparison apples to apples?** Same grain, same horizon, same 216 keys. `assert_same_rows` raises if any model's keys differ.

**What happens with a new region?** `OneHotEncoder(handle_unknown="ignore")` gives all zeros; the lags still work if there are 12 months of history. Otherwise you'd fall back to a baseline.

**What is WAPE's weakness?** Big series dominate it. That is why I report per-region WAPE, where North's best model is the seasonal naive.

## K-Means
**Why standardize?** K-Means uses Euclidean distance. Without scaling, units (thousands) would swamp discount % (single digits).

**What is a centroid?** The mean of a cluster's points, here shown in z-scores to compare features.

**How did you choose K?** I scored K = 2–4 with silhouette, required at least 2 cells per cluster, checked seed stability (ARI = 1.0 across 10 seeds), and preferred the smaller K when silhouettes were within 0.01. K = 3 (0.448) and K = 4 (0.449) were tied, so K = 3.

**Why are 18 cells too few?** Each cluster has only a handful of points; one changed cell can change the solution, and silhouette is noisy.

**Can segments tell you which region has higher marketing ROI?** No. They describe past behavior; ROI needs intervention evidence.

## SHAP
**Local vs global?** Local: why this row got this prediction (waterfall). Global: mean |SHAP| across many rows.

**Why can correlated lags distort the ranking?** If `lag_12` and `roll_mean_12` carry similar information, trees may split on either, and credit is shared arbitrarily.

**Why isn't SHAP causal?** It explains the model's function, not the data-generating process. Changing spend in the real world also changes things the model never saw.

## Optimization
**Decision variable?** Spend per region, `b_r`, in ₹ crore.

**Constraints?** `Σ b_r = B` and `min_r ≤ b_r ≤ max_r`.

**Why diminishing returns?** Saturation is the standard assumption: each extra crore reaches less responsive customers. `log(1 + b/scale)` is concave, so the optimum is global.

**Where did the parameters come from?** Hand-set for illustration, and documented in `docs/assumptions.md`. They were not estimated from the data.

**Infeasible budget?** If B is below Σ min or above Σ max, the code raises `InfeasibleBudgetError` and the dashboard shows the feasible range.

**Why compare with equal allocation?** It is a simple, defensible default. The optimizer must beat it under the stated assumptions, or it isn't worth the complexity. Here it is +₹4.3 cr (1.8%) at ₹100 cr.

**What experiment would you run first?** A geo-randomized spend test (matched regions or territories with increased and held spend) to estimate real response curves.

## Engineering
**Can someone reproduce this?** Yes: seed 42, `n_jobs=1`, the clean CSV SHA-256 is in metadata, version ranges are pinned, and a README one-command flow. Every score is written by code.

**What did you debug?** Prophet 1.1.7 failed on Windows because cmdstanpy 1.3 requires a `makefile` the bundled CmdStan lacks. I pinned `cmdstanpy<1.3`.

**Why Streamlit, not microservices?** One user, local artifacts, no write traffic. A web API would add deployment cost and no value.
