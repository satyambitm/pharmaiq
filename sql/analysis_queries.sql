-- PharmaIQ Lite+ analytical queries (synthetic data; all findings illustrative).
-- Each query starts with "-- name:" so src/analytics/sql_reporting.py can run it by name.

-- name: q01_monthly_totals
-- question: What were total units and net revenue each month?
SELECT month_start, units_sold, net_revenue_inr
FROM v_monthly_totals
ORDER BY month_start;

-- name: q02_mom_revenue_growth
-- question: How did net revenue change month over month? NULLIF avoids divide-by-zero.
WITH m AS (
    SELECT month_start, net_revenue_inr,
           LAG(net_revenue_inr) OVER (ORDER BY month_start) AS prev_net_revenue_inr
    FROM v_monthly_totals
)
SELECT month_start, net_revenue_inr, prev_net_revenue_inr,
       ROUND(100.0 * (net_revenue_inr - prev_net_revenue_inr) / NULLIF(prev_net_revenue_inr, 0), 2) AS mom_growth_pct
FROM m
ORDER BY month_start;

-- name: q03_region_revenue_share
-- question: What share of total net revenue does each region contribute (full 60 months)?
WITH r AS (
    SELECT region_id, region_name, SUM(net_revenue_inr) AS net_revenue_inr
    FROM v_region_month
    GROUP BY region_id, region_name
)
SELECT region_id, region_name, net_revenue_inr,
       ROUND(100.0 * net_revenue_inr / SUM(net_revenue_inr) OVER (), 2) AS revenue_share_pct
FROM r
ORDER BY net_revenue_inr DESC;

-- name: q04_product_rank_by_region
-- question: How do products rank by net revenue within each region? DENSE_RANK gives tied
-- products the same rank without skipping the next rank; ROW_NUMBER would break ties arbitrarily.
WITH rp AS (
    SELECT region_name, product_name, SUM(net_revenue_inr) AS net_revenue_inr
    FROM v_sales_enriched
    GROUP BY region_name, product_name
)
SELECT region_name, product_name, net_revenue_inr,
       DENSE_RANK() OVER (PARTITION BY region_name ORDER BY net_revenue_inr DESC) AS product_rank
FROM rp
ORDER BY region_name, product_rank;

-- name: q05_region_3m_moving_avg
-- question: What is each region's trailing 3-month moving average of net revenue?
-- The frame includes the current month and the two before it (first two months use fewer rows).
SELECT month_start, region_name, net_revenue_inr,
       ROUND(AVG(net_revenue_inr) OVER (
           PARTITION BY region_id ORDER BY month_start
           ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) AS net_revenue_3m_ma,
       COUNT(*) OVER (
           PARTITION BY region_id ORDER BY month_start
           ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS months_in_window
FROM v_region_month
ORDER BY region_name, month_start;

-- name: q06_yoy_units_growth
-- question: How did units change versus the same month last year (per region x product)?
-- LAG(..., 12) is valid only because the panel is complete (checked by q10).
WITH s AS (
    SELECT month_start, region_name, product_name, units_sold,
           LAG(units_sold, 12) OVER (PARTITION BY region_name, product_name ORDER BY month_start) AS units_same_month_ly
    FROM v_sales_enriched
)
SELECT month_start, region_name, product_name, units_sold, units_same_month_ly,
       ROUND(100.0 * (units_sold - units_same_month_ly) / NULLIF(units_same_month_ly, 0), 2) AS yoy_growth_pct
FROM s
WHERE units_same_month_ly IS NOT NULL
ORDER BY region_name, product_name, month_start;

-- name: q07_campaign_vs_noncampaign
-- question: Descriptively, how do campaign months compare with non-campaign months?
-- NONCAUSAL: campaigns are not randomized and differ in spend; this is not campaign uplift.
SELECT product_name, campaign_flag,
       COUNT(*) AS n_rows,
       ROUND(AVG(units_sold), 1) AS avg_units,
       ROUND(AVG(marketing_spend_inr), 2) AS avg_spend_inr,
       ROUND(AVG(100.0 * discount_inr / NULLIF(gross_revenue_inr, 0)), 2) AS avg_discount_pct
FROM v_sales_enriched
GROUP BY product_name, campaign_flag
ORDER BY product_name, campaign_flag;

-- name: q08_price_discount_distribution
-- question: How are unit price and discount rate distributed per product and year?
SELECT product_name, EXTRACT(YEAR FROM month_start)::INT AS year,
       ROUND(MIN(unit_price_inr), 2) AS min_price_inr,
       ROUND(AVG(unit_price_inr), 2) AS avg_price_inr,
       ROUND(MAX(unit_price_inr), 2) AS max_price_inr,
       ROUND(100.0 * SUM(discount_inr) / NULLIF(SUM(gross_revenue_inr), 0), 2) AS weighted_discount_pct,
       ROUND(CORR(unit_price_inr::float8, units_sold::float8)::numeric, 3) AS corr_price_units
FROM v_sales_enriched
GROUP BY product_name, EXTRACT(YEAR FROM month_start)
ORDER BY product_name, year;

-- name: q09_region_volatility
-- question: Which regions have the most volatile monthly units? Coefficient of variation
-- (stddev / mean) makes regions of different size comparable. Uses all 60 complete months.
SELECT region_name,
       COUNT(*) AS n_months,
       ROUND(AVG(units_sold), 1) AS mean_units,
       ROUND(STDDEV_SAMP(units_sold), 1) AS sd_units,
       ROUND(STDDEV_SAMP(units_sold) / NULLIF(AVG(units_sold), 0), 4) AS cv_units
FROM v_region_month
GROUP BY region_name
ORDER BY cv_units DESC;

-- name: q10_missing_panel_keys
-- question: Are any (month, region, product) combinations missing? Expect zero rows.
WITH calendar AS (
    SELECT generate_series(DATE '2021-01-01', DATE '2025-12-01', INTERVAL '1 month')::date AS month_start
)
SELECT c.month_start, r.region_id, p.product_id
FROM calendar c
CROSS JOIN regions r
CROSS JOIN products p
LEFT JOIN monthly_sales s
  ON s.month_start = c.month_start AND s.region_id = r.region_id AND s.product_id = p.product_id
WHERE s.month_start IS NULL
ORDER BY 1, 2, 3;
