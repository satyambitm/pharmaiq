"""Write example upload files for the Data Management page and the ingestion tests.

Usage: python scripts/make_byod_samples.py

All three files are synthetic (seeded) and use different column names and grains:
* territory_monthly_sales.csv      monthly, 6 territories x 2 drugs x 30 months, price + promotion + competitor
                                   price + HCP calls, with a few deliberate defects (negative quantity,
                                   bad date, duplicated row) so the quality report has something to show
* pharmacy_transactions_daily.csv  daily transactions, DD/MM/YYYY dates, revenue but no price or marketing
* region_units_only.csv            monthly units only, no product column (single-product dataset), 14 months
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "byod_examples"
TERRITORIES = ["Mumbai West", "Pune", "Delhi NCR", "Lucknow", "Chennai", "Kolkata"]


def territory_monthly(rng: np.random.Generator) -> pd.DataFrame:
    months = pd.date_range("2023-07-01", periods=30, freq="MS")
    rows = []
    for t_i, territory in enumerate(TERRITORIES):
        for drug, base, price in [("Glucofix 500", 4200, 38.5), ("Cardiolin 10", 2600, 74.0)]:
            level = base * (0.7 + 0.12 * t_i)
            for i, m in enumerate(months):
                season = 1 + 0.08 * np.sin(2 * np.pi * (m.month - 3) / 12)
                promo = round(float(rng.uniform(1.5, 4.0) * level), 2)
                units = level * season * (1 + 0.006 * i) * (1 + promo / level / 60) * rng.normal(1, 0.05)
                rows.append({"transaction_month": m.strftime("%Y-%m"), "territory": territory, "drug_name": drug,
                             "sales_quantity": int(round(units)), "selling_price": round(price * rng.uniform(0.97, 1.03), 2),
                             "promotion_cost": promo, "rival_price": round(price * rng.uniform(0.9, 1.1), 2),
                             "hcp_calls": int(rng.integers(80, 220))})
    df = pd.DataFrame(rows)
    df.loc[5, "sales_quantity"] = -120          # negative quantity -> rejected
    df.loc[17, "transaction_month"] = "2024-13"  # impossible month -> rejected
    return pd.concat([df, df.iloc[[40]]], ignore_index=True)  # one exact duplicate row


def pharmacy_daily(rng: np.random.Generator) -> pd.DataFrame:
    days = pd.date_range("2024-03-04", "2025-08-27", freq="D")
    rows = []
    for city, scale in [("Bengaluru", 1.3), ("Hyderabad", 1.0), ("Jaipur", 0.7), ("Kochi", 0.5)]:
        for brand, mrp in [("Painex", 12.0), ("Allercet", 6.5), ("Gastrozol", 18.0)]:
            for d in days:
                if rng.random() < 0.55:
                    qty = int(rng.poisson(30 * scale))
                    rows.append({"Date": d.strftime("%d/%m/%Y"), "City": city, "Brand": brand, "Qty": qty,
                                 "Net Amount": f"{qty * mrp * rng.uniform(0.92, 1.0):,.2f}"})
    return pd.DataFrame(rows)


def region_units(rng: np.random.Generator) -> pd.DataFrame:
    months = pd.date_range("2024-09-01", periods=14, freq="MS")
    return pd.DataFrame([{"Month": m.strftime("%b %Y"), "Region": r, "Units": int(rng.normal(900, 60) * (1 + j / 5))}
                         for j, r in enumerate(["North", "South", "East", "West"]) for m in months])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20251103)
    for name, df in [("territory_monthly_sales.csv", territory_monthly(rng)),
                     ("pharmacy_transactions_daily.csv", pharmacy_daily(rng)),
                     ("region_units_only.csv", region_units(rng))]:
        df.to_csv(OUT / name, index=False, lineterminator="\n")
        print(f"{name}: {len(df)} rows")


if __name__ == "__main__":
    main()
