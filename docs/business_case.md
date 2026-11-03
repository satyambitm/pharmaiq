# Business case — Aster Pharma (fictional)

> All numbers come from generated artifacts (`artifacts/metrics/eda_summary.json`, `artifacts/reports/sql_*.csv`, `artifacts/metrics/model_comparison.csv`, `artifacts/clusters/*`, `artifacts/metrics/budget_*.csv`). The data is **synthetic**; conclusions are illustrative.

## Context
Aster Pharma sells Therapy A, B and C in six regions. Leadership wants: historical performance, a next-month forecast, descriptive market segments, and a way to explore marketing budget splits.

## Five EDA findings

**1. Steady growth, with revenue growing faster than units.**
*Analysis:* yearly totals. Units rose from 617,467 (2021) to 826,884 (2025), a 7.6% CAGR; net revenue rose from ₹73.9 cr to ₹120.3 cr.
*Interpretation:* revenue grows faster because of the annual 4% list-price increase built into the generator.
*Recommendation:* track price and volume separately in reporting.
*Caveat:* growth rates are a generator parameter.

**2. Revenue is moderately concentrated in three regions.**
*Analysis:* SQL q03. North 21.8%, South 19.7%, West 19.6%, East 16.2%, Central 12.7%, Northeast 10.1%.
*Interpretation:* the top three regions produce about 61% of net revenue, so they dominate any volume-weighted metric like WAPE.
*Recommendation:* always report per-region forecast accuracy as well (done in the model card).

**3. Therapy B has the strongest seasonality.**
*Analysis:* calendar-month profile. Seasonal amplitude (max − min of the monthly average / mean) is 44% for Therapy B vs 22% (A) and 24% (C). Peak months: A in April, B in August, C in October.
*Interpretation:* B needs seasonal stock and target planning; a flat monthly target would mislead.
*Caveat:* the monthly profile also includes trend, which slightly inflates amplitude.

**4. Smaller regions are more volatile.**
*Analysis:* SQL q09. Coefficient of variation of regional monthly units ranges from 0.117 (East) to 0.158 (Northeast). The most volatile series is Northeast / Therapy B (CV 0.245); the least is South / Therapy A (0.112).
*Interpretation:* small, highly seasonal cells will have larger relative forecast errors.

**5. Campaign months show higher units, but that is not campaign uplift.**
*Analysis:* SQL q07. Average units in campaign vs non-campaign months: A 5,178 vs 4,836; B 3,533 vs 3,355; C 1,891 vs 1,767. Campaign months also carry about 55–70% more spend and about 3 percentage points higher discount.
*Interpretation:* campaign months bundle extra spend, extra discount and a campaign flag, so this comparison cannot separate their effects. The pooled spend–units correlation (0.41) mostly reflects that bigger cells both spend more and sell more.
*Recommendation:* use a randomized or geo-holdout test before attributing uplift.

## Forecasting
On the untouched 2025 holdout (216 identical rows), Prophet (WAPE 7.88%) and XGBoost (8.06%) both beat seasonal naive (9.79%) and last-month naive (9.58%). XGBoost was better on every 2023–2024 backtest fold. Per region, the best model varies; North was best served by the seasonal naive.
*Recommendation:* treat XGBoost and Prophet as roughly comparable here. XGBoost is about 200× cheaper to run, which makes it the practical default; keep the seasonal naive as a monitored floor.

## Segmentation
K-Means on 18 market cells (history through Dec 2023) chose K = 3. The three clusters are exactly the three products:
- *High volume, stable, slower growth* — Therapy A in all regions
- *Mid volume, volatile* — Therapy B (strong seasonality, highest discount)
- *Low volume, moderate variability* — Therapy C (high spend per unit, low discount)

*Interpretation:* product-level characteristics (seasonality, discount, price-driven spend per unit) dominate the standardized descriptors; regional differences are mostly scale. That is the honest answer for this generator: segmenting by product already captures the structure.
*Caveat:* N = 18; descriptive only; no ROI interpretation. Labels describe cluster centroids, so a member can differ. For example, Northeast / Therapy A is in the "high volume" cluster despite being smaller than some Therapy B cells.

## Budget simulator (assumed response curves)
For a ₹100 cr budget, under the base assumptions, the optimized split yields an *assumed* ₹237.7 cr incremental revenue vs ₹233.4 cr for an equal split (+₹4.3 cr, about 1.8%). The assumed status-quo split gets ₹237.0 cr. With 30% competitor pressure on North and West, the optimizer moves budget away from them and the advantage over equal split shrinks to ₹1.5 cr.
*Interpretation:* with concave curves, the gain from optimizing is modest once every region is above its minimum. The bigger decisions are the bounds and the response assumptions themselves.
*Recommendation:* before using any allocation, run a geo-experiment to estimate real response curves.
