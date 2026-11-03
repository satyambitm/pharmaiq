-- Reusable reporting views. Safe to re-run.

CREATE OR REPLACE VIEW v_sales_enriched AS
SELECT
    s.month_start,
    s.region_id,
    r.region_name,
    s.product_id,
    p.product_name,
    s.units_sold,
    s.unit_price_inr,
    s.gross_revenue_inr,
    s.discount_inr,
    s.net_revenue_inr,
    s.marketing_spend_inr,
    s.active_accounts,
    s.competitor_index,
    s.regional_demand_index,
    s.campaign_flag,
    s.data_source
FROM monthly_sales s
JOIN regions  r ON r.region_id  = s.region_id
JOIN products p ON p.product_id = s.product_id;

CREATE OR REPLACE VIEW v_monthly_totals AS
SELECT
    month_start,
    SUM(units_sold)          AS units_sold,
    SUM(gross_revenue_inr)   AS gross_revenue_inr,
    SUM(discount_inr)        AS discount_inr,
    SUM(net_revenue_inr)     AS net_revenue_inr,
    SUM(marketing_spend_inr) AS marketing_spend_inr
FROM monthly_sales
GROUP BY month_start;

CREATE OR REPLACE VIEW v_region_month AS
SELECT
    s.month_start,
    s.region_id,
    r.region_name,
    SUM(s.units_sold)          AS units_sold,
    SUM(s.net_revenue_inr)     AS net_revenue_inr,
    SUM(s.marketing_spend_inr) AS marketing_spend_inr
FROM monthly_sales s
JOIN regions r ON r.region_id = s.region_id
GROUP BY s.month_start, s.region_id, r.region_name;
