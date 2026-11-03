"""Descriptive K-Means over a dataset's region x product cells (same approach as the demo's segmentation).

Descriptors are used only when every cell has them, so nothing is imputed. Segments summarise history;
they say nothing about causes or marketing ROI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

KEY = ["region", "product"]
MIN_CELLS, MIN_REGIONS, MAX_K = 6, 3, 5


def cell_descriptors(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (region, product), g in rows.groupby(KEY, sort=True):
        g = g.sort_values("month_start")
        units = g["units_sold"].astype(float)
        d = {"region": region, "product": product, "months_observed": int(len(g)),
             "avg_monthly_units": units.mean(),
             "cv_units": units.std() / units.mean() if len(units) > 1 and units.mean() > 0 else np.nan}
        if len(g) >= 12:
            recent, prior = units.iloc[-6:].mean(), units.iloc[-12:-6].mean()
            d["recent_trend_pct"] = (recent / prior - 1) * 100 if prior > 0 else np.nan
        if len(g) >= 24:
            profile = g.groupby(g["month_start"].dt.month)["units_sold"].mean()
            d["seasonality_amplitude"] = (profile.max() - profile.min()) / profile.mean() if profile.mean() > 0 else np.nan
        if g["revenue"].notna().all() and units.sum() > 0:
            d["avg_unit_price"] = g["revenue"].sum() / units.sum()
        if g["marketing_spend"].notna().all() and units.sum() > 0:
            d["marketing_per_unit"] = g["marketing_spend"].sum() / units.sum()
        out.append(d)
    return pd.DataFrame(out)


def _labels(centers_z: pd.DataFrame) -> dict[int, str]:
    names = {}
    for c, row in centers_z.iterrows():
        vol = "high volume" if row["avg_monthly_units"] > 0.5 else "low volume" if row["avg_monthly_units"] < -0.5 else "mid volume"
        var = ("" if "cv_units" not in row else ", volatile" if row["cv_units"] > 0.5
               else ", stable" if row["cv_units"] < -0.5 else "")
        trend = ("" if "recent_trend_pct" not in row else ", growing" if row["recent_trend_pct"] > 0.5
                 else ", declining" if row["recent_trend_pct"] < -0.5 else "")
        names[int(c)] = f"{vol}{var}{trend}"
    counts: dict[str, int] = {}
    for c in sorted(names):
        if list(names.values()).count(names[c]) > 1:
            counts[names[c]] = counts.get(names[c], 0) + 1
            names[c] = f"{names[c]} (#{counts[names[c]]})"
    return names


def run(rows: pd.DataFrame, seed: int) -> dict:
    cells = cell_descriptors(rows)
    n_regions = int(cells["region"].nunique())
    if len(cells) < MIN_CELLS or n_regions < MIN_REGIONS:
        return {"status": "skipped", "reason": f"Needs at least {MIN_REGIONS} regions and {MIN_CELLS} region/product cells; "
                                               f"this dataset has {n_regions} and {len(cells)}."}
    candidates = ["avg_monthly_units", "cv_units", "recent_trend_pct", "seasonality_amplitude", "avg_unit_price",
                  "marketing_per_unit"]
    used = [c for c in candidates if c in cells and cells[c].notna().all()]
    omitted = [c for c in candidates if c not in used]
    X = StandardScaler().fit_transform(cells[used])
    table = []
    for k in range(2, min(MAX_K, len(cells) - 1) + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
        table.append({"k": k, "silhouette": round(float(silhouette_score(X, km.labels_)), 4),
                      "min_cluster_size": int(np.bincount(km.labels_).min())})
    ok = [t for t in table if t["min_cluster_size"] >= 2] or table
    best = max(t["silhouette"] for t in ok)
    k = min(t["k"] for t in ok if t["silhouette"] >= best - 0.01)
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
    centers = pd.DataFrame(km.cluster_centers_, columns=used)
    names = _labels(centers)
    cells["cluster"] = km.labels_
    cells["segment_label"] = cells["cluster"].map(names)
    profiles = (cells.groupby(["cluster", "segment_label"])[used].mean().round(4).reset_index()
                .assign(n_cells=cells.groupby("cluster").size().to_numpy()))
    for t in table:
        t["chosen"] = t["k"] == k
    return {
        "status": "ok", "k": k, "features_used": used, "features_omitted": omitted, "k_selection": table,
        "cells": cells.round(4).astype(object).where(cells.notna(), None).to_dict("records"),
        "profiles": profiles.to_dict("records"),
        "note": f"Exploratory clustering of {len(cells)} cells; segments describe past sales patterns only.",
    }
