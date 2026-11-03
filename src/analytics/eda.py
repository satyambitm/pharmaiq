"""EDA figures and summary statistics exported from the clean panel (synthetic data)."""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config

logger = logging.getLogger(__name__)
CR = 1e7  # 1 crore = 10,000,000 INR
SUBTITLE = "Synthetic demo data (synthetic_v1)"


def _save(fig: plt.Figure, name: str) -> None:
    fig.text(0.99, 0.005, SUBTITLE, ha="right", va="bottom", fontsize=7, color="grey")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, dpi=120)
    plt.close(fig)


def enrich(panel: pd.DataFrame) -> pd.DataFrame:
    """Add readable names and calendar columns."""
    df = panel.copy()
    df["region_name"] = df["region_id"].map(config.REGIONS)
    df["product_name"] = df["product_id"].map(config.PRODUCTS)
    df["year"] = df["month_start"].dt.year
    df["month"] = df["month_start"].dt.month
    return df


def make_figures(panel: pd.DataFrame) -> list[str]:
    """Create the EDA plot checklist; return file names written."""
    config.ensure_dirs()
    df = enrich(panel)
    monthly = df.groupby("month_start")[["units_sold", "net_revenue_inr"]].sum()
    written = []

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly.index, monthly["units_sold"], color="#2E86AB")
    ax.set(title="Total monthly units sold (all regions, products)", xlabel="Month", ylabel="Units")
    _save(fig, "eda_01_monthly_units.png"); written.append("eda_01_monthly_units.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly.index, monthly["net_revenue_inr"] / CR, color="#A23B72")
    ax.set(title="Total monthly net revenue", xlabel="Month", ylabel="Net revenue (INR crore)")
    _save(fig, "eda_02_monthly_net_revenue.png"); written.append("eda_02_monthly_net_revenue.png")

    reg = df.groupby("region_name")["net_revenue_inr"].sum().sort_values() / CR
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(reg.index, reg.values, color="#F18F01")
    ax.set(title="Net revenue by region, Jan 2021 - Dec 2025", xlabel="Net revenue (INR crore)", ylabel="Region")
    _save(fig, "eda_03_revenue_by_region.png"); written.append("eda_03_revenue_by_region.png")

    share = df.pivot_table(index="region_name", columns="product_name", values="net_revenue_inr", aggfunc="sum")
    share = share / share.to_numpy().sum() * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(share, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax, cbar_kws={"label": "% of total net revenue"})
    ax.set(title="Region x product share of total net revenue (%)", xlabel="Product", ylabel="Region")
    _save(fig, "eda_04_region_product_heatmap.png"); written.append("eda_04_region_product_heatmap.png")

    prof = df.groupby(["product_name", "month"])["units_sold"].sum().reset_index()
    prof["index_vs_mean"] = prof.groupby("product_name")["units_sold"].transform(lambda s: s / s.mean())
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.lineplot(prof, x="month", y="index_vs_mean", hue="product_name", marker="o", ax=ax)
    ax.axhline(1, color="grey", lw=0.8, ls="--")
    ax.set(title="Calendar-month seasonality profile (units / product average)", xlabel="Calendar month", ylabel="Seasonal index")
    ax.set_xticks(range(1, 13))
    _save(fig, "eda_05_seasonality_profile.png"); written.append("eda_05_seasonality_profile.png")

    roll = monthly["units_sold"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(roll.index, roll, alpha=0.4, label="monthly units")
    ax.plot(roll.index, roll.rolling(12).mean(), label="12-month rolling mean")
    ax2 = ax.twinx()
    ax2.plot(roll.index, roll.rolling(12).std(), color="red", alpha=0.6, label="12-month rolling std")
    ax.set(title="Total units: rolling mean and volatility", xlabel="Month", ylabel="Units")
    ax2.set_ylabel("Rolling std (units)")
    ax.legend(loc="upper left"); ax2.legend(loc="lower right")
    _save(fig, "eda_06_rolling_mean_volatility.png"); written.append("eda_06_rolling_mean_volatility.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(df, x=df["marketing_spend_inr"] / 1e5, y="units_sold", hue="product_name", alpha=0.6, ax=ax)
    ax.set(title="Marketing spend vs units (descriptive only, not causal)", xlabel="Marketing spend (INR lakh)", ylabel="Units sold")
    _save(fig, "eda_07_spend_vs_units.png"); written.append("eda_07_spend_vs_units.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(df, x="region_name", y="units_sold", hue="product_name", ax=ax)
    ax.set(title="Distribution of monthly units per region x product", xlabel="Region", ylabel="Units sold per month")
    _save(fig, "eda_08_units_boxplot.png"); written.append("eda_08_units_boxplot.png")

    s = df.sort_values(["region_id", "product_id", "month_start"])
    g = s.groupby(["region_id", "product_id"])["units_sold"]
    lagged = pd.DataFrame({"units": s["units_sold"], **{f"lag_{k}": g.shift(k) for k in (1, 2, 3, 6, 12)},
                           "spend_lag_1": s.groupby(["region_id", "product_id"])["marketing_spend_inr"].shift(1),
                           "competitor_lag_1": s.groupby(["region_id", "product_id"])["competitor_index"].shift(1)}).dropna()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(lagged.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation of units with lagged features (pooled across series)")
    _save(fig, "eda_09_lag_correlation.png"); written.append("eda_09_lag_correlation.png")
    return written


def summary_stats(panel: pd.DataFrame) -> dict:
    """Numbers that back the documented EDA findings (all generated, none hand-typed)."""
    df = enrich(panel)
    yearly = df.groupby("year")[["units_sold", "net_revenue_inr"]].sum()
    reg_share = (df.groupby("region_name")["net_revenue_inr"].sum() / df["net_revenue_inr"].sum() * 100).round(2)
    prod_share = (df.groupby("product_name")["net_revenue_inr"].sum() / df["net_revenue_inr"].sum() * 100).round(2)
    series_cv = df.groupby(["region_name", "product_name"])["units_sold"].agg(lambda s: s.std() / s.mean())
    prof = df.groupby(["product_name", "month"])["units_sold"].mean()
    amp = prof.groupby(level=0).agg(lambda s: (s.max() - s.min()) / s.mean() * 100).round(1)
    peak = prof.groupby(level=0).idxmax().map(lambda t: int(t[1]))
    cam = df.groupby("campaign_flag")["units_sold"].mean()
    return {
        "rows": int(len(df)),
        "months": int(df["month_start"].nunique()),
        "yearly_units": {int(k): int(v) for k, v in yearly["units_sold"].items()},
        "yearly_net_revenue_cr": {int(k): round(v / CR, 2) for k, v in yearly["net_revenue_inr"].items()},
        "units_cagr_2021_2025_pct": round(((yearly["units_sold"].iloc[-1] / yearly["units_sold"].iloc[0]) ** (1 / 4) - 1) * 100, 2),
        "region_revenue_share_pct": reg_share.to_dict(),
        "product_revenue_share_pct": prod_share.to_dict(),
        "most_volatile_series": series_cv.idxmax(), "most_volatile_cv": round(float(series_cv.max()), 3),
        "least_volatile_series": series_cv.idxmin(), "least_volatile_cv": round(float(series_cv.min()), 3),
        "seasonal_amplitude_pct_by_product": amp.to_dict(),
        "peak_calendar_month_by_product": peak.to_dict(),
        "avg_units_campaign_vs_not": {"campaign": round(float(cam.get(True, np.nan)), 1), "no_campaign": round(float(cam.get(False, np.nan)), 1)},
        "corr_spend_units_pooled": round(float(df["marketing_spend_inr"].corr(df["units_sold"])), 3),
        "weighted_discount_pct": round(float(df["discount_inr"].sum() / df["gross_revenue_inr"].sum() * 100), 2),
        "zero_unit_rows": int((df["units_sold"] == 0).sum()),
    }


def run_eda(panel: pd.DataFrame) -> dict:
    """Write figures + eda_summary.json; return the summary."""
    files = make_figures(panel)
    stats = summary_stats(panel)
    stats["figures"] = files
    (config.METRICS_DIR / "eda_summary.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    logger.info("EDA: %d figures written", len(files))
    return stats
