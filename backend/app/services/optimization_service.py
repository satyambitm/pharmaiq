"""Scenario-based budget allocation under ASSUMED response curves (see src/optimization/response.py)."""

from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np

from backend.app.core.errors import InvalidRequest
from src.optimization.allocate import InfeasibleBudgetError, budget_sensitivity, compare_strategies
from src.optimization.response import DEFAULT_RESPONSES, apply_scenario

LABEL = "Scenario-based simulation under assumed response curves - not a measured ROI or causal estimate."
ASSUMPTIONS = [
    "Each region's gain follows alpha * ln(1 + spend / scale); alpha and scale are hand-set illustrative inputs.",
    "The curves were not estimated from the sales history, SHAP or correlations.",
    "Scenario low/base/high multiplies every alpha by 0.8 / 1.0 / 1.2; this changes total gain, not the optimal split.",
    "Competitor pressure p shrinks that region's alpha by a factor (1 - p).",
    "Equal split and status quo can break the regional min/max bounds; see within_bounds.",
]
SENSITIVITY_POINTS = 15


def optimize_budget(total_budget_cr: float, scenario: str, competitor_pressure: dict[str, float],
                    min_spend_cr: float = 5.0, max_spend_cr: float = 30.0) -> dict:
    bounded = tuple(replace(r, min_spend=min_spend_cr, max_spend=max_spend_cr) for r in DEFAULT_RESPONSES)
    responses = apply_scenario(bounded, scenario, competitor_pressure)
    try:
        table = compare_strategies(total_budget_cr, responses)
    except InfeasibleBudgetError as exc:
        raise InvalidRequest(str(exc)) from exc

    strategies, totals = [], {}
    for name, g in table.groupby("strategy", sort=False):
        totals[name] = float(g["assumed_gain_cr"].sum())
        strategies.append({
            "strategy": name, "total_assumed_gain_cr": round(totals[name], 3),
            "feasible": bool(g["within_bounds"].all()),
            "allocation": [{"region": r.region, "spend_cr": round(r.spend_cr, 3),
                            "assumed_gain_cr": round(r.assumed_gain_cr, 3),
                            "marginal_gain_per_cr": round(r.marginal_gain_per_cr, 4),
                            "min_cr": r.min_cr, "max_cr": r.max_cr, "within_bounds": bool(r.within_bounds)}
                           for r in g.itertuples()],
        })
    lo, hi = sum(r.min_spend for r in responses), sum(r.max_spend for r in responses)
    sens = budget_sensitivity(list(np.linspace(lo, hi, SENSITIVITY_POINTS)), responses)
    return {
        "label": LABEL, "assumptions": ASSUMPTIONS, "total_budget_cr": total_budget_cr, "scenario": scenario,
        "feasible_budget_range_cr": (lo, hi),
        "response_curves": [{k: round(v, 4) if isinstance(v, float) else v for k, v in asdict(r).items()} for r in responses],
        "strategies": strategies,
        "optimized_vs_equal_split_cr": round(totals["optimized"] - totals["equal_split"], 3),
        "optimized_vs_status_quo_cr": round(totals["optimized"] - totals["status_quo"], 3),
        "sensitivity": sens.round(3).to_dict("records"),
    }
