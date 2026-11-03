"""Constrained budget allocation under the assumed concave response curves (SLSQP).

maximize  sum_r alpha_r * ln(1 + b_r / scale_r)
s.t.      sum_r b_r = B,   min_r <= b_r <= max_r

The objective is concave and the feasible set is convex, so a local optimum is global.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.optimization.response import DEFAULT_RESPONSES, RegionResponse, gain, marginal_gain

logger = logging.getLogger(__name__)
TOL = 1e-6


class InfeasibleBudgetError(ValueError):
    """Raised when the budget cannot satisfy the regional min/max constraints."""


def _arrays(responses: tuple[RegionResponse, ...]) -> dict[str, np.ndarray]:
    return {
        "alpha": np.array([r.alpha for r in responses]),
        "scale": np.array([r.scale for r in responses]),
        "lo": np.array([r.min_spend for r in responses]),
        "hi": np.array([r.max_spend for r in responses]),
    }


def check_feasible(budget: float, responses: tuple[RegionResponse, ...] = DEFAULT_RESPONSES) -> None:
    """Raise InfeasibleBudgetError unless sum(min) <= budget <= sum(max)."""
    a = _arrays(responses)
    if np.any(a["lo"] > a["hi"]):
        raise InfeasibleBudgetError("a region has min spend greater than max spend")
    lo, hi = float(a["lo"].sum()), float(a["hi"].sum())
    if not lo - TOL <= budget <= hi + TOL:
        raise InfeasibleBudgetError(f"budget {budget:.2f} cr is outside feasible range [{lo:.2f}, {hi:.2f}] cr")


def feasible_start(budget: float, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Feasible starting point: fill each region's headroom proportionally."""
    room = hi - lo
    return lo + (budget - lo.sum()) * room / room.sum() if room.sum() > 0 else lo.copy()


def optimize(budget: float, responses: tuple[RegionResponse, ...] = DEFAULT_RESPONSES) -> pd.DataFrame:
    """Optimal allocation table; asserts solver success, budget conservation and bounds."""
    check_feasible(budget, responses)
    a = _arrays(responses)
    x0 = feasible_start(budget, a["lo"], a["hi"])
    res = minimize(
        lambda b: -gain(b, a["alpha"], a["scale"]).sum(),
        x0,
        jac=lambda b: -marginal_gain(b, a["alpha"], a["scale"]),
        bounds=list(zip(a["lo"], a["hi"])),
        constraints=[{"type": "eq", "fun": lambda b: b.sum() - budget, "jac": lambda b: np.ones_like(b)}],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"optimizer failed: {res.message}")
    b = np.clip(res.x, a["lo"], a["hi"])
    if abs(b.sum() - budget) > 1e-4:
        raise RuntimeError(f"budget not conserved: allocated {b.sum():.6f} vs {budget:.6f}")
    return allocation_table(b, responses, "optimized")


def allocation_table(spend: np.ndarray, responses: tuple[RegionResponse, ...], strategy: str) -> pd.DataFrame:
    """Per-region spend, assumed gain, marginal gain, and constraint flags for an allocation."""
    a = _arrays(responses)
    spend = np.asarray(spend, float)
    g = gain(spend, a["alpha"], a["scale"])
    return pd.DataFrame({
        "strategy": strategy,
        "region": [r.region for r in responses],
        "spend_cr": spend,
        "assumed_gain_cr": g,
        "assumed_gain_per_cr": np.divide(g, spend, out=np.zeros_like(g), where=spend > 0),
        "marginal_gain_per_cr": marginal_gain(spend, a["alpha"], a["scale"]),
        "min_cr": a["lo"],
        "max_cr": a["hi"],
        "within_bounds": (spend >= a["lo"] - 1e-6) & (spend <= a["hi"] + 1e-6),
    })


def equal_split(budget: float, responses: tuple[RegionResponse, ...] = DEFAULT_RESPONSES) -> pd.DataFrame:
    """B / n per region (may violate bounds; see `within_bounds`)."""
    return allocation_table(np.full(len(responses), budget / len(responses)), responses, "equal_split")


def status_quo(budget: float, responses: tuple[RegionResponse, ...] = DEFAULT_RESPONSES) -> pd.DataFrame:
    """Assumed existing allocation shares scaled to B (may violate bounds; see `within_bounds`)."""
    shares = np.array([r.status_quo_share for r in responses])
    return allocation_table(budget * shares / shares.sum(), responses, "status_quo")


def compare_strategies(budget: float, responses: tuple[RegionResponse, ...] = DEFAULT_RESPONSES) -> pd.DataFrame:
    """Stacked table of optimized, equal-split and status-quo allocations."""
    return pd.concat([optimize(budget, responses), equal_split(budget, responses), status_quo(budget, responses)],
                     ignore_index=True)


def budget_sensitivity(budgets: list[float], responses: tuple[RegionResponse, ...] = DEFAULT_RESPONSES) -> pd.DataFrame:
    """Total assumed gain for optimized vs equal split across feasible budgets."""
    rows = []
    for b in budgets:
        try:
            opt = optimize(b, responses)
        except InfeasibleBudgetError:
            continue
        eq = equal_split(b, responses)
        rows.append({"budget_cr": b, "optimized_gain_cr": opt["assumed_gain_cr"].sum(),
                     "equal_split_gain_cr": eq["assumed_gain_cr"].sum(),
                     "equal_split_feasible": bool(eq["within_bounds"].all())})
    return pd.DataFrame(rows)
