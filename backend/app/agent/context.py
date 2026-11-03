"""The data an agent conversation may read, fixed by the server before the model runs.

A context wraps exactly one dataset: the synthetic demo, or one uploaded dataset already checked to belong to the
caller. Tools receive only the context, so the model can never name a different dataset. Every method is a read
that returns JSON-safe facts taken from stored data or evaluated model results; when something cannot be answered
the method says so instead of estimating.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from backend.app.core.errors import AppError
from backend.app.db.connection import SalesRepository
from backend.app.schemas.common import SYNTHETIC_NOTE
from backend.app.services import (analytics_service, forecast_service, insights_service, optimization_service,
                                  segmentation_service)
from ml.model_loader import ModelBundle
from src.config import PRODUCTS, REGIONS

SCENARIOS = ("low", "base", "high")


class ToolInputError(ValueError):
    """Bad arguments from the model; the message is returned to it so it can correct itself."""


def _month(value: str | None, name: str) -> date | None:
    if not value:
        return None
    try:
        return pd.Period(value, freq="M").start_time.date()
    except (ValueError, TypeError):
        raise ToolInputError(f"{name} must look like YYYY-MM, got {value!r}") from None


def _pick(value: str, options: dict[str, str], what: str) -> str:
    """Match a code or a display name, case-insensitively, to a key of `options`."""
    v = value.strip().lower()
    for key, label in options.items():
        if v in (key.lower(), label.lower()):
            return key
    raise ToolInputError(f"Unknown {what} {value!r}. Valid: {', '.join(options.values())}")


def _tool_error(exc: AppError) -> ToolInputError:
    return ToolInputError(exc.message)


class DemoContext:
    kind = "demo"

    def __init__(self, repo: SalesRepository, bundle: ModelBundle) -> None:
        self.repo, self.bundle = repo, bundle
        self.label = "Synthetic demo dataset (fictional company, 6 regions x 3 products, monthly 2021-2025)"

    def describe(self) -> str:
        return (f"{self.label}. Regions: {', '.join(REGIONS.values())}. Products: {', '.join(PRODUCTS.values())}. "
                "Money is in INR crore (1 crore = 10 million). " + SYNTHETIC_NOTE)

    def sales_summary(self, start_month: str | None, end_month: str | None, region: str | None) -> dict:
        start, end = _month(start_month, "start_month"), _month(end_month, "end_month")
        try:
            ov = analytics_service.overview(self.repo, start, end)
            rg = analytics_service.regions(self.repo, start, end)
        except AppError as exc:
            raise _tool_error(exc) from exc
        out = {"window_start": str(ov["window_start"]), "window_end": str(ov["window_end"]), "months": ov["months"],
               "money_unit": "INR crore"}
        if region:
            rid = _pick(region, REGIONS, "region")
            out["region"] = next(r for r in rg["regions"] if r["region_id"] == rid)
            out["last_6_months"] = [{"month": str(m["month"]), "net_revenue_cr": round(m["net_revenue_cr"], 2),
                                     "units_sold": m["units_sold"]} for m in rg["monthly"] if m["region_id"] == rid][-6:]
        else:
            out["kpis"] = ov["kpis"]
            out["regions"] = rg["regions"]
            out["product_mix"] = ov["product_mix"]
        return out

    def sales_forecast(self, region: str, horizon: int, model: str | None, product: str | None) -> dict:
        rid = _pick(region, REGIONS, "region")
        model = model or "xgboost"
        if model not in ("xgboost", "prophet"):
            raise ToolInputError("model must be 'xgboost' or 'prophet'")
        try:
            fc = forecast_service.forecast(self.repo, self.bundle, rid, horizon, model)
        except AppError as exc:
            raise _tool_error(exc) from exc
        rows = fc["forecast"]
        if product:
            pid = _pick(product, PRODUCTS, "product")
            rows = [r for r in rows if r["product_id"] == pid]
        return {"region": fc["region_name"], "model": model, "model_version": fc["model_version"],
                "forecast_origin_month": str(fc["origin_month"]), "units": "units sold per month",
                "forecast": [{**r, "target_month": str(r["target_month"])} for r in rows],
                "interval": fc["interval"], "backtest_accuracy_2025": fc["backtest_accuracy_2025"],
                "assumptions": fc["assumptions"]}

    def market_segments(self) -> dict:
        s = segmentation_service.segments(self.bundle)
        return {"method": "K-Means on region x product cells", "k": s["k"], "features": s["features"],
                "history_cutoff": s["history_cutoff"], "profiles": s["profiles"], "cells": s["cells"],
                "notes": s["notes"]}

    def explain_forecast(self) -> dict:
        ins = insights_service.model_insights(self.bundle)
        return {"method": "SHAP values of the XGBoost forecaster", "global_importance": ins["global_importance"][:12],
                "caveat": ins["shap_caveat"], "evaluation": ins["evaluation"],
                "horizon_scorecard": ins["horizon_scorecard"]}

    def compare_budget_scenarios(self, total_budget_cr: float, scenarios: list[str] | None,
                                 competitor_pressure: dict[str, float] | None) -> dict:
        names = scenarios or list(SCENARIOS)
        if any(s not in SCENARIOS for s in names):
            raise ToolInputError("scenarios must be a subset of low, base, high")
        pressure = {}
        for k, v in (competitor_pressure or {}).items():
            if not 0 <= v < 1:
                raise ToolInputError("competitor pressure values must be in [0, 1)")
            pressure[REGIONS[_pick(k, REGIONS, "region")]] = v
        results = []
        for sc in names:
            try:
                r = optimization_service.optimize_budget(total_budget_cr, sc, pressure)
            except AppError as exc:
                raise _tool_error(exc) from exc
            opt = next(s for s in r["strategies"] if s["strategy"] == "optimized")
            results.append({"scenario": sc,
                            "total_assumed_gain_cr": {s["strategy"]: s["total_assumed_gain_cr"] for s in r["strategies"]},
                            "optimized_vs_equal_split_cr": r["optimized_vs_equal_split_cr"],
                            "optimized_allocation": [{"region": a["region"], "spend_cr": a["spend_cr"]}
                                                     for a in opt["allocation"]]})
        return {"label": optimization_service.LABEL, "assumptions": optimization_service.ASSUMPTIONS,
                "total_budget_cr": total_budget_cr, "competitor_pressure": pressure, "scenarios": results}

    def data_quality(self) -> dict:
        df = self.repo.fetch_sales()
        missing = {c: int(n) for c, n in df.isna().sum().items() if n}
        return {"dataset": self.label, "source": "synthetic", "disclaimer": SYNTHETIC_NOTE,
                "monthly_rows": int(len(df)), "window_start": str(df["month_start"].min().date()),
                "window_end": str(df["month_start"].max().date()), "regions": sorted(df["region_id"].map(REGIONS).unique()),
                "products": sorted(df["product_id"].map(PRODUCTS).unique()), "missing_values": missing,
                "available_analyses": {"revenue": True, "marketing": True, "budget_simulation": True}}


class UploadContext:
    kind = "upload"

    def __init__(self, meta: dict, rows: pd.DataFrame, version: dict | None) -> None:
        self.meta, self.rows, self.version = meta, rows, version
        self.report = meta.get("quality_report") or {}
        self.caps = self.report.get("capabilities", {})
        self.label = f"Uploaded dataset '{meta['dataset_name']}'"
        self.regions = {r: r for r in sorted(rows["region"].unique())}
        self.products = {p: p for p in sorted(rows["product"].unique())}

    def _enabled(self, cap: str) -> bool:
        return bool(self.caps.get(cap, {}).get("enabled"))

    def describe(self) -> str:
        cur = self.report.get("currency") or "an undeclared currency"
        return (f"{self.label}, monthly {self.meta['date_start']} to {self.meta['date_end']}. "
                f"Regions: {', '.join(self.regions)}. Products: {', '.join(self.products)}. "
                f"Revenue analysis {'available' if self._enabled('revenue_analysis') else 'NOT available'}; "
                f"marketing analysis {'available' if self._enabled('marketing_analysis') else 'NOT available'}. "
                f"Money is in {cur}. "
                + ("Models have been trained." if self.version else "No models have been trained on it yet."))

    def _no_models(self) -> dict:
        return {"available": False, "reason": "No models have been trained on this dataset yet. The owner can train "
                                              "them on the Data Management page."}

    def sales_summary(self, start_month: str | None, end_month: str | None, region: str | None) -> dict:
        start, end = _month(start_month, "start_month"), _month(end_month, "end_month")
        df = self.rows
        if start:
            df = df[df["month_start"] >= pd.Timestamp(start)]
        if end:
            df = df[df["month_start"] <= pd.Timestamp(end)]
        if region:
            df = df[df["region"] == _pick(region, self.regions, "region")]
        if df.empty:
            raise ToolInputError("No rows in the selected window")
        revenue = self._enabled("revenue_analysis") and df["revenue"].notna().all()
        marketing = self._enabled("marketing_analysis") and df["marketing_spend"].notna().all()

        def agg(g: pd.DataFrame) -> dict:
            d = {"units_sold": round(float(g["units_sold"].sum()), 2)}
            if revenue:
                d["revenue"] = round(float(g["revenue"].sum()), 2)
            if marketing:
                d["marketing_spend"] = round(float(g["marketing_spend"].sum()), 2)
            return d

        monthly = df.groupby("month_start")["units_sold"].sum().sort_index()
        growth = None
        if len(monthly) >= 24 and monthly.iloc[-24:-12].sum() > 0:
            growth = round(float((monthly.iloc[-12:].sum() / monthly.iloc[-24:-12].sum() - 1) * 100), 2)
        out = {"window_start": str(df["month_start"].min().date()), "window_end": str(df["month_start"].max().date()),
               "months": int(monthly.size), "totals": agg(df), "units_growth_pct_last12_vs_prior12": growth,
               "by_region": {r: agg(g) for r, g in df.groupby("region")},
               "by_product": {p: agg(g) for p, g in df.groupby("product")},
               "last_6_months_units": {str(k.date()): round(float(v), 2) for k, v in monthly.iloc[-6:].items()},
               "currency": self.report.get("currency")}
        if not revenue:
            out["revenue_note"] = self.caps.get("revenue_analysis", {}).get("reason") or "Revenue is incomplete."
        if not marketing:
            out["marketing_note"] = self.caps.get("marketing_analysis", {}).get("reason") or "Marketing spend is incomplete."
        if growth is None:
            out["growth_note"] = "Needs 24 months of data in the window."
        return out

    def sales_forecast(self, region: str, horizon: int, model: str | None, product: str | None) -> dict:
        if not self.version:
            return self._no_models()
        fc = self.version["results"]["forecasting"]
        model = model or fc["recommended_model"]
        rid = _pick(region, self.regions, "region")
        rows = [f for f in fc["forecasts"] if f["model"] == model and f["region"] == rid and f["step"] <= horizon]
        if product:
            pid = _pick(product, self.products, "product")
            rows = [f for f in rows if f["product"] == pid]
        if not rows:
            raise ToolInputError(f"No {model} forecast for that selection. Models with forecasts: "
                                 f"{sorted({f['model'] for f in fc['forecasts']})}")
        return {"region": rid, "model": model, "recommended_model": fc["recommended_model"],
                "model_version": self.version["version_id"], "trained_at": str(self.version["created_at"]),
                "units": "units sold per month",
                "forecast": [{k: f[k] for k in ("product", "target_month", "step", "y_pred")} for f in rows],
                "holdout_evaluation": fc["evaluation"], "notes": fc["notes"] + [fc["forecast_note"]]}

    def market_segments(self) -> dict:
        if not self.version:
            return self._no_models()
        seg = self.version["results"]["segmentation"]
        if seg["status"] != "ok":
            return {"available": False, "reason": seg["reason"]}
        return {"method": "K-Means on region x product cells", "k": seg["k"], "features": seg["features_used"],
                "profiles": seg["profiles"], "cells": [{k: c.get(k) for k in ("region", "product", "segment_label")}
                                                       for c in seg["cells"]], "notes": [seg["note"]]}

    def explain_forecast(self) -> dict:
        if not self.version:
            return self._no_models()
        shap = self.version["results"]["explainability"]
        if shap["status"] != "ok":
            return {"available": False, "reason": shap["reason"], "caveat": shap["caveat"]}
        return {"method": shap["method"], "global_importance": shap["global_importance"][:12], "caveat": shap["caveat"],
                "model_version": self.version["version_id"]}

    def compare_budget_scenarios(self, total_budget_cr: float, scenarios: list[str] | None,
                                 competitor_pressure: dict[str, float] | None) -> dict:
        return {"available": False,
                "reason": "Budget scenarios use hand-set response curves defined only for the synthetic demo regions. "
                          "There is no defensible response model for uploaded data, so no allocation or gain is "
                          "computed. Marketing effects would need experiments or a causal study."}

    def data_quality(self) -> dict:
        r = self.report
        return {"dataset": self.label, "status": self.meta["status"], "file_rows": r.get("total_rows"),
                "valid_rows": r.get("valid_rows"), "rejected_rows": r.get("rejected_rows"),
                "rejection_reasons": r.get("rejection_reasons"), "monthly_rows": self.meta.get("row_count"),
                "detected_frequency": r.get("detected_frequency"), "coverage": r.get("coverage"),
                "missing_value_pct": r.get("missing_value_pct"), "warnings": r.get("warnings"),
                "forecast_readiness": r.get("forecast_readiness"), "capabilities": self.caps,
                "model_features": r.get("model_features"),
                "trained_model_version": self.version["version_id"] if self.version else None}
