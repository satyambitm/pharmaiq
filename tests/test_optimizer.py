import numpy as np
import pandas as pd
import pytest

from src.optimization import allocate
from src.optimization.response import DEFAULT_RESPONSES, apply_scenario


def test_optimizer_budget_constraints():
    for budget in (30.0, 100.0, 180.0):
        t = allocate.optimize(budget)
        assert np.isfinite(t["spend_cr"]).all()
        assert t["spend_cr"].sum() == pytest.approx(budget, abs=1e-4)
        assert t["within_bounds"].all()


def test_optimizer_beats_or_matches_equal_split():
    opt = allocate.optimize(100.0)["assumed_gain_cr"].sum()
    eq = allocate.equal_split(100.0)["assumed_gain_cr"].sum()
    assert opt >= eq - 1e-9


def test_optimizer_rejects_infeasible_budget():
    with pytest.raises(allocate.InfeasibleBudgetError):
        allocate.optimize(10.0)
    with pytest.raises(allocate.InfeasibleBudgetError):
        allocate.optimize(500.0)


def test_optimizer_is_deterministic_and_scale_invariant():
    a = allocate.optimize(100.0)
    pd.testing.assert_frame_equal(a, allocate.optimize(100.0))
    high = allocate.optimize(100.0, apply_scenario(DEFAULT_RESPONSES, "high"))
    assert np.allclose(a["spend_cr"], high["spend_cr"], atol=1e-3)
    pressured = allocate.optimize(100.0, apply_scenario(DEFAULT_RESPONSES, "base", {"North": 0.5}))
    north = lambda t: t.loc[t["region"] == "North", "spend_cr"].iloc[0]  # noqa: E731
    assert north(pressured) < north(a)
