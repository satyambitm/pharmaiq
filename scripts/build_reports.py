"""EDA figures, SQL report exports, K-Means segments and budget scenarios.

Usage: python scripts/build_reports.py [--skip-db]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.analytics import eda, sql_reporting  # noqa: E402
from src.data.database import db_available  # noqa: E402
from src.optimization import allocate, response  # noqa: E402
from src.segmentation import kmeans  # noqa: E402

logger = logging.getLogger("build_reports")
DEFAULT_BUDGET_CR = 100.0


def budget_scenarios() -> None:
    """Strategy comparison per scenario + budget sensitivity, saved as CSV and a figure."""
    tables, summary = [], []
    scenarios = {
        "low": response.apply_scenario(response.DEFAULT_RESPONSES, "low"),
        "base": response.apply_scenario(response.DEFAULT_RESPONSES, "base"),
        "high": response.apply_scenario(response.DEFAULT_RESPONSES, "high"),
        "base_competitor_pressure_north_west_30pct": response.apply_scenario(
            response.DEFAULT_RESPONSES, "base", {"North": 0.3, "West": 0.3}),
    }
    for name, resp in scenarios.items():
        t = allocate.compare_strategies(DEFAULT_BUDGET_CR, resp).assign(scenario=name)
        tables.append(t)
        tot = t.groupby("strategy")["assumed_gain_cr"].sum()
        summary.append({"scenario": name, "budget_cr": DEFAULT_BUDGET_CR, **{f"{k}_gain_cr": round(v, 3) for k, v in tot.items()},
                        "optimized_minus_equal_cr": round(tot["optimized"] - tot["equal_split"], 3)})
    pd.concat(tables, ignore_index=True).to_csv(config.METRICS_DIR / "budget_allocations.csv", index=False)
    pd.DataFrame(summary).to_csv(config.METRICS_DIR / "budget_scenario_summary.csv", index=False)

    sens = allocate.budget_sensitivity(list(np.arange(30, 181, 10.0)), scenarios["base"])
    sens.to_csv(config.METRICS_DIR / "budget_sensitivity.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sens["budget_cr"], sens["optimized_gain_cr"], "o-", label="optimized")
    ax.plot(sens["budget_cr"], sens["equal_split_gain_cr"], "s--", label="equal split")
    ax.set(title="ASSUMED incremental revenue vs budget (illustrative, not causal ROI)",
           xlabel="Total budget (INR crore)", ylabel="Assumed incremental revenue (INR crore)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "budget_sensitivity.png", dpi=120)
    plt.close(fig)
    logger.info("Budget scenarios:\n%s", pd.DataFrame(summary).to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-db", action="store_true", help="skip SQL report exports")
    args = parser.parse_args()
    config.setup_logging()
    config.ensure_dirs()
    panel = pd.read_parquet(config.PANEL_PARQUET)

    eda.run_eda(panel)

    if not args.skip_db and db_available():
        counts = sql_reporting.export_all()
        checks = sql_reporting.reconcile_with_pandas(panel)
        (config.METRICS_DIR / "sql_reconciliation.json").write_text(json.dumps({"row_counts": counts, "checks": checks}, indent=2), encoding="utf-8")
        logger.info("SQL vs pandas checks: %s", checks)
        if not all(checks.values()):
            logger.error("SQL reconciliation failed")
            return 1
    else:
        logger.warning("Skipping SQL exports (database unavailable or --skip-db)")

    seg = kmeans.run_segmentation(panel)
    kmeans.save_artifacts(seg)
    logger.info("K selection:\n%s", seg["k_table"].to_string(index=False))
    logger.info("Profiles:\n%s", seg["profiles"].to_string(index=False))

    budget_scenarios()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
