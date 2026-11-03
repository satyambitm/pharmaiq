"""ASSUMED diminishing-returns response curves for the budget simulator.

    gain_r(b) = alpha_r * ln(1 + b / scale_r)        (b, alpha, scale all in INR crore)

These parameters are illustrative inputs chosen by hand. They were NOT estimated from the
synthetic history, from SHAP, or from correlations, and they are not measured causal uplift.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class RegionResponse:
    """Assumed response parameters and spend bounds for one region (INR crore)."""

    region: str
    alpha: float
    scale: float
    min_spend: float
    max_spend: float
    status_quo_share: float


DEFAULT_RESPONSES: tuple[RegionResponse, ...] = (
    RegionResponse("North", alpha=60.0, scale=15.0, min_spend=5.0, max_spend=30.0, status_quo_share=0.24),
    RegionResponse("South", alpha=45.0, scale=12.0, min_spend=5.0, max_spend=30.0, status_quo_share=0.20),
    RegionResponse("East", alpha=40.0, scale=10.0, min_spend=5.0, max_spend=30.0, status_quo_share=0.17),
    RegionResponse("West", alpha=55.0, scale=14.0, min_spend=5.0, max_spend=30.0, status_quo_share=0.19),
    RegionResponse("Central", alpha=30.0, scale=8.0, min_spend=5.0, max_spend=30.0, status_quo_share=0.12),
    RegionResponse("Northeast", alpha=25.0, scale=6.0, min_spend=5.0, max_spend=30.0, status_quo_share=0.08),
)

SCENARIO_ALPHA_MULTIPLIER = {"low": 0.8, "base": 1.0, "high": 1.2}


def gain(spend: np.ndarray, alpha: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Per-region assumed incremental revenue (INR crore)."""
    return alpha * np.log1p(np.asarray(spend, float) / scale)


def marginal_gain(spend: np.ndarray, alpha: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """d gain / d spend: assumed incremental revenue per extra crore at the current spend."""
    return alpha / (scale + np.asarray(spend, float))


def apply_scenario(responses: tuple[RegionResponse, ...], scenario: str = "base",
                   competitor_pressure: dict[str, float] | None = None) -> tuple[RegionResponse, ...]:
    """Scale alphas by the scenario multiplier and optional per-region competitor pressure.

    competitor_pressure maps region -> fraction in [0, 1) by which that region's alpha shrinks.
    A uniform multiplier on every alpha changes total gain but not the optimal allocation.
    """
    mult = SCENARIO_ALPHA_MULTIPLIER[scenario]
    pressure = competitor_pressure or {}
    out = []
    for r in responses:
        p = pressure.get(r.region, 0.0)
        if not 0.0 <= p < 1.0:
            raise ValueError(f"competitor pressure for {r.region} must be in [0, 1)")
        out.append(replace(r, alpha=r.alpha * mult * (1 - p)))
    return tuple(out)
