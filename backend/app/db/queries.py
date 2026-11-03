"""Parameterized SQL used by the API. Values are always bound, never string-formatted."""

from sqlalchemy import text

PING = text("SELECT 1")

COUNT_ROWS = text("SELECT COUNT(*) FROM monthly_sales WHERE data_source = :data_source")

# Optional filters use the "parameter IS NULL OR ..." pattern so one statement serves all cases.
SALES_ROWS = text("""
    SELECT month_start, region_id, product_id, units_sold, unit_price_inr, gross_revenue_inr,
           discount_inr, net_revenue_inr, marketing_spend_inr, active_accounts,
           competitor_index, regional_demand_index, campaign_flag
    FROM monthly_sales
    WHERE data_source = :data_source
      AND (CAST(:start_date AS DATE) IS NULL OR month_start >= CAST(:start_date AS DATE))
      AND (CAST(:end_date AS DATE)   IS NULL OR month_start <= CAST(:end_date AS DATE))
      AND (CAST(:region_ids AS TEXT[])  IS NULL OR region_id  = ANY(CAST(:region_ids AS TEXT[])))
      AND (CAST(:product_ids AS TEXT[]) IS NULL OR product_id = ANY(CAST(:product_ids AS TEXT[])))
    ORDER BY region_id, product_id, month_start
""")

DATA_WINDOW = text("""
    SELECT MIN(month_start) AS first_month, MAX(month_start) AS last_month, COUNT(*) AS n_rows
    FROM monthly_sales WHERE data_source = :data_source
""")
