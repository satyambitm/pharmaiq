-- PharmaIQ Lite+ schema (synthetic_v1). Safe to re-run.
-- Grain of monthly_sales: one row per (month_start, region_id, product_id).

CREATE TABLE IF NOT EXISTS regions (
    region_id   TEXT PRIMARY KEY,
    region_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS products (
    product_id        TEXT PRIMARY KEY,
    product_name      TEXT NOT NULL UNIQUE,
    therapeutic_group TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS monthly_sales (
    month_start           DATE          NOT NULL,
    region_id             TEXT          NOT NULL REFERENCES regions(region_id),
    product_id            TEXT          NOT NULL REFERENCES products(product_id),
    units_sold            INTEGER       NOT NULL CHECK (units_sold >= 0),
    unit_price_inr        NUMERIC(14,2) NOT NULL CHECK (unit_price_inr > 0),
    gross_revenue_inr     NUMERIC(14,2) NOT NULL CHECK (gross_revenue_inr >= 0),
    discount_inr          NUMERIC(14,2) NOT NULL CHECK (discount_inr >= 0),
    net_revenue_inr       NUMERIC(14,2) NOT NULL CHECK (net_revenue_inr >= 0),
    marketing_spend_inr   NUMERIC(14,2) NOT NULL CHECK (marketing_spend_inr >= 0),
    active_accounts       INTEGER       NOT NULL CHECK (active_accounts >= 0),
    competitor_index      DOUBLE PRECISION NOT NULL,
    regional_demand_index DOUBLE PRECISION NOT NULL,
    campaign_flag         BOOLEAN       NOT NULL,
    data_source           TEXT          NOT NULL DEFAULT 'synthetic_v1',
    PRIMARY KEY (month_start, region_id, product_id),
    CONSTRAINT chk_month_start CHECK (EXTRACT(DAY FROM month_start) = 1),
    CONSTRAINT chk_discount_le_gross CHECK (discount_inr <= gross_revenue_inr),
    -- NUMERIC arithmetic is exact, so reconciliation needs no tolerance.
    CONSTRAINT chk_gross_reconciles CHECK (gross_revenue_inr = units_sold * unit_price_inr),
    CONSTRAINT chk_net_reconciles CHECK (net_revenue_inr = gross_revenue_inr - discount_inr)
);

CREATE INDEX IF NOT EXISTS idx_monthly_sales_region_month ON monthly_sales (region_id, month_start);
CREATE INDEX IF NOT EXISTS idx_monthly_sales_product_month ON monthly_sales (product_id, month_start);
