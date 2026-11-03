"""Deterministic synthetic monthly sales generator (dataset version `synthetic_v1`).

Grain: one row per (month_start, region_id, product_id); 60 months x 6 regions x 3 products.

Generating equation (EQUATION_VERSION = gen-eq-1.0), for series (r, p) at month index t:

    units = base_p * region_mult_r
            * (1 + (trend_p + trend_r) * t)                       # linear trend
            * (1 + amp_p * sin(2*pi*(month - phase_p) / 12))      # annual seasonality
            * regional_demand_index[r, t]                         # AR(1) around 1.0
            * (1 - 0.25 * (competitor_index[r, p, t] - 1))        # competitor pressure
            * (1 + 0.12 * (1 - exp(-exposure_t / base_spend_rp))) # capped, concave marketing
            * (1 + 0.05 * campaign_flag)                          # mild campaign lift
            * sqrt(active_accounts / expected_accounts)           # coverage
            * lognormal(0, sigma_p)                               # noise

    exposure_t = 0.6 * spend[t-1] + 0.4 * spend[t]   (lagged marketing exposure)

Marketing spend is a plan generated independently of units (seasonal budget, campaign bump,
noise), so there is no backdoor from current units to spend. All relationships are *assumed*
by construction; recovering them with a model is not evidence about real markets.

Money is computed in integer paise so that gross = units * price and
net = gross - discount reconcile exactly to 2 decimals.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

PRODUCT_PARAMS: dict[str, dict[str, float]] = {
    # base units/month, list price INR (2021), monthly trend, seasonal amp/phase, noise sigma, base discount
    "P01": {"base": 4000, "price": 850.0, "trend": 0.004, "amp": 0.12, "phase": 1, "sigma": 0.05, "disc": 0.08},
    "P02": {"base": 2500, "price": 1400.0, "trend": 0.008, "amp": 0.20, "phase": 4, "sigma": 0.07, "disc": 0.12},
    "P03": {"base": 1200, "price": 2600.0, "trend": 0.012, "amp": 0.08, "phase": 7, "sigma": 0.10, "disc": 0.05},
}
REGION_PARAMS: dict[str, dict[str, float]] = {
    "R01": {"mult": 1.30, "trend": 0.000},
    "R02": {"mult": 1.15, "trend": 0.002},
    "R03": {"mult": 1.00, "trend": -0.001},
    "R04": {"mult": 1.10, "trend": 0.003},
    "R05": {"mult": 0.75, "trend": 0.001},
    "R06": {"mult": 0.55, "trend": 0.004},
}
SPEND_SHARE_OF_REVENUE = 0.08
CAMPAIGN_PROB = 0.15
ANNUAL_PRICE_INCREASE = 0.04

CSV_COLUMNS = [
    "month_start", "region_id", "product_id", "units_sold", "unit_price_inr",
    "gross_revenue_inr", "discount_inr", "net_revenue_inr", "marketing_spend_inr",
    "active_accounts", "competitor_index", "regional_demand_index", "campaign_flag", "data_source",
]
MONEY_COLUMNS = ["unit_price_inr", "gross_revenue_inr", "discount_inr", "net_revenue_inr", "marketing_spend_inr"]


def _ar1(rng: np.random.Generator, n: int, phi: float, sd: float, lo: float, hi: float) -> np.ndarray:
    """AR(1) path around 1.0, clipped to [lo, hi]."""
    x = np.empty(n)
    prev = 1.0
    for i in range(n):
        prev = 1.0 + phi * (prev - 1.0) + rng.normal(0.0, sd)
        prev = float(np.clip(prev, lo, hi))
        x[i] = prev
    return x


def _paise_to_str(paise: np.ndarray) -> list[str]:
    """Format integer paise as exact 2-decimal INR strings (avoids float rounding)."""
    return [f"{'-' if p < 0 else ''}{abs(int(p)) // 100}.{abs(int(p)) % 100:02d}" for p in paise]


def generate_clean(seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Generate the clean, complete rectangular panel.

    Returns a DataFrame with money columns as exact Decimal-friendly strings in INR.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range(config.START_MONTH, periods=config.N_MONTHS, freq="MS")
    t = np.arange(config.N_MONTHS)
    cal_month = months.month.to_numpy()
    years_since_start = (months.year - config.START_MONTH.year).to_numpy()

    demand = {r: _ar1(rng, config.N_MONTHS, phi=0.6, sd=0.03, lo=0.85, hi=1.15) for r in config.REGIONS}

    frames: list[pd.DataFrame] = []
    for r, rp in REGION_PARAMS.items():
        for p, pp in PRODUCT_PARAMS.items():
            expected_units = pp["base"] * rp["mult"]
            base_spend_inr = expected_units * pp["price"] * SPEND_SHARE_OF_REVENUE

            competitor = _ar1(rng, config.N_MONTHS, phi=0.8, sd=0.05, lo=0.6, hi=1.4)
            campaign = rng.random(config.N_MONTHS) < CAMPAIGN_PROB
            spend = (
                base_spend_inr
                * (1 + 0.15 * np.cos(2 * np.pi * (cal_month - 2) / 12))
                * (1 + 0.6 * campaign)
                * rng.lognormal(0.0, 0.15, config.N_MONTHS)
            )
            spend_prev = np.concatenate([[spend[0]], spend[:-1]])
            exposure = 0.6 * spend_prev + 0.4 * spend

            expected_accounts = np.round(expected_units / 25 * (1 + 0.003 * t))
            accounts = np.round(expected_accounts * rng.normal(1.0, 0.02, config.N_MONTHS)).astype(int)
            accounts = np.maximum(accounts, 1)

            units_mean = (
                expected_units
                * (1 + (pp["trend"] + rp["trend"]) * t)
                * (1 + pp["amp"] * np.sin(2 * np.pi * (cal_month - pp["phase"]) / 12))
                * demand[r]
                * (1 - 0.25 * (competitor - 1))
                * (1 + 0.12 * (1 - np.exp(-exposure / base_spend_inr)))
                * (1 + 0.05 * campaign)
                * np.sqrt(accounts / expected_accounts)
            )
            units = np.maximum(0, np.round(units_mean * rng.lognormal(0.0, pp["sigma"], config.N_MONTHS))).astype(np.int64)

            list_price = pp["price"] * (1 + ANNUAL_PRICE_INCREASE) ** years_since_start
            price_paise = np.round(list_price * rng.normal(1.0, 0.01, config.N_MONTHS) * 100).astype(np.int64)
            gross_paise = units * price_paise
            disc_rate = np.clip(pp["disc"] + 0.03 * campaign + rng.normal(0, 0.01, config.N_MONTHS), 0.0, 0.30)
            disc_paise = np.round(gross_paise * disc_rate).astype(np.int64)
            net_paise = gross_paise - disc_paise
            spend_paise = np.round(spend * 100).astype(np.int64)

            frames.append(pd.DataFrame({
                "month_start": months.strftime("%Y-%m-%d"),
                "region_id": r,
                "product_id": p,
                "units_sold": units,
                "unit_price_inr": _paise_to_str(price_paise),
                "gross_revenue_inr": _paise_to_str(gross_paise),
                "discount_inr": _paise_to_str(disc_paise),
                "net_revenue_inr": _paise_to_str(net_paise),
                "marketing_spend_inr": _paise_to_str(spend_paise),
                "active_accounts": accounts,
                "competitor_index": np.round(competitor, 4),
                "regional_demand_index": np.round(demand[r], 4),
                "campaign_flag": campaign,
                "data_source": config.DATA_SOURCE,
            }))

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["month_start", "region_id", "product_id"]).reset_index(drop=True)
    return df[CSV_COLUMNS]


def make_dirty(clean: pd.DataFrame, seed: int = config.RANDOM_SEED) -> tuple[pd.DataFrame, list[dict]]:
    """Inject controlled anomalies into a *copy* of the clean panel and log each one."""
    rng = np.random.default_rng(seed + 1)
    dirty = clean.copy()
    log: list[dict] = []
    n = len(dirty)
    used: set[int] = set()

    def pick(k: int) -> list[int]:
        choices = [i for i in rng.permutation(n) if i not in used][:k]
        used.update(choices)
        return [int(i) for i in choices]

    def key(i: int) -> dict:
        row = clean.loc[i]
        return {"month_start": row.month_start, "region_id": row.region_id, "product_id": row.product_id}

    for i in pick(max(1, round(0.01 * n))):
        dirty.loc[i, "competitor_index"] = np.nan
        log.append({"type": "missing_competitor_index", **key(i)})
    for i in pick(3):
        dirty.loc[i, "marketing_spend_inr"] = "-" + str(dirty.loc[i, "marketing_spend_inr"])
        log.append({"type": "negative_marketing_spend", **key(i)})
    for i in pick(3):
        dirty.loc[i, "net_revenue_inr"] = f"{float(dirty.loc[i, 'net_revenue_inr']) + 1234.56:.2f}"
        log.append({"type": "net_revenue_mismatch", **key(i)})
    for i in pick(2):
        dirty.loc[i, "month_start"] = str(dirty.loc[i, "month_start"])[:8] + "15"
        log.append({"type": "mid_month_date", **key(i)})
    dropped = pick(2)
    for i in dropped:
        log.append({"type": "missing_panel_row", **key(i)})
    dup_idx = pick(5)
    for i in dup_idx:
        log.append({"type": "duplicate_key", **key(i)})

    dirty = pd.concat([dirty.drop(index=dropped), dirty.loc[dup_idx]], ignore_index=True)
    return dirty, log


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes, recorded in metadata for reproducibility checks."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_outputs(seed: int = config.RANDOM_SEED) -> dict:
    """Generate clean + dirty CSVs and metadata JSON; return the metadata dict."""
    config.ensure_dirs()
    clean = generate_clean(seed)
    dirty, anomalies = make_dirty(clean, seed)
    clean.to_csv(config.CLEAN_CSV, index=False, lineterminator="\n")
    dirty.to_csv(config.DIRTY_CSV, index=False, lineterminator="\n")

    metadata = {
        "dataset_version": config.DATA_SOURCE,
        "equation_version": config.EQUATION_VERSION,
        "seed": seed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "grain": "month_start x region_id x product_id",
        "coverage": {"start": str(config.START_MONTH.date()), "months": config.N_MONTHS},
        "n_regions": len(config.REGIONS),
        "n_products": len(config.PRODUCTS),
        "n_rows_clean": len(clean),
        "n_rows_dirty": len(dirty),
        "clean_sha256": file_sha256(config.CLEAN_CSV),
        "price_convention": "unit_price_inr = average realized gross (pre-discount) price per unit; "
                            "list price rises 4% each January with ~1% monthly noise",
        "competitor_index_scale": "AR(1) around 1.0, clipped [0.6, 1.4]; 1.0 = typical pressure, higher = more pressure",
        "regional_demand_index_scale": "AR(1) around 1.0 per region, clipped [0.85, 1.15]",
        "product_params": PRODUCT_PARAMS,
        "region_params": REGION_PARAMS,
        "spend_share_of_expected_revenue": SPEND_SHARE_OF_REVENUE,
        "campaign_probability": CAMPAIGN_PROB,
        "anomalies": anomalies,
        "disclaimer": "Synthetic, illustrative data. Not real market or causal marketing evidence.",
    }
    config.METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Clean rows=%d, dirty rows=%d, anomalies=%d, sha256=%s",
                len(clean), len(dirty), len(anomalies), metadata["clean_sha256"][:12])
    return metadata


def main() -> None:
    config.setup_logging()
    write_outputs()


if __name__ == "__main__":
    main()
