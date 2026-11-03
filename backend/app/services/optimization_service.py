"""Scenario-based budget allocation under ASSUMED response curves (see src/optimization/response.py)."""

from __future__ import annotations

from backend.app.core.errors import InvalidRequest
from src.optimization.allocate import InfeasibleBudgetError, compare_strategies
from src.optimization.response import DEFAULT_RESPONSES, apply_scenario

LABEL = "Scenario-based simulation under assumed response curves - not a measured ROI or causal estimate."
ASSUMPTIONS = [
    "Each region's gain follows alpha * ln(1 + spend / scale); alpha and scale are hand-set illustrative inputs.",
    "The curves were not estimated from the sales history, SHAP or correlations.",
    "Scenario low/base/high multiplies every alpha by 0.8 / 1.0 / 1.2; this changes total gain, not the optimal split.",
    "Competitor pressure p shrinks that region's alpha by a factor (1 - p).",
    "Equal split and status quo can break the regional min/max bounds; see within_bounds.",
]


def optimize_budget(total_budget_cr: float, scenario: str, competitor_pressure: dict[str, float]) -> dict:
    responses = apply_scenario(DEFAULT_RESPONSES, scenario, competitor_pressure)
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
    return {
        "label": LABEL, "assumptions": ASSUMPTIONS, "total_budget_cr": total_budget_cr, "scenario": scenario,
        "feasible_budget_range_cr": (sum(r.min_spend for r in responses), sum(r.max_spend for r in responses)),
        "strategies": strategies,
        "optimized_vs_equal_split_cr": round(totals["optimized"] - totals["equal_split"], 3),
        "optimized_vs_status_quo_cr": round(totals["optimized"] - totals["status_quo"], 3),
    }
