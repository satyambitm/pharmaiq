"""Descriptive K-Means segmentation of the 18 region x product market cells.

Descriptors are computed from history up to SEGMENTATION_CUTOFF (end of 2023) only, then
standardized and clustered for K = 2..4. With N = 18 the segments are exploratory: they
summarize historical behavior and say nothing about causal drivers or marketing ROI.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src import config

logger = logging.getLogger(__name__)

SEGMENT_FEATURES = [
    "avg_monthly_units", "cv_units", "yoy_growth_pct", "avg_discount_pct",
    "spend_per_unit_inr", "seasonality_amplitude", "campaign_frequency",
]
K_RANGE = range(2, 5)
STABILITY_SEEDS = range(10)


def market_cell_features(panel: pd.DataFrame, cutoff: pd.Timestamp = config.SEGMENTATION_CUTOFF) -> pd.DataFrame:
    """One row per (region, product) using only months <= cutoff."""
    hist = panel[panel["month_start"] <= cutoff].copy()
    hist["year"] = hist["month_start"].dt.year
    last_year = cutoff.year
    rows = []
    for (r, p), g in hist.groupby(["region_id", "product_id"]):
        units = g["units_sold"].astype(float)
        by_year = g.groupby("year")["units_sold"].sum()
        prev = by_year.get(last_year - 1, np.nan)
        cur = by_year.get(last_year, np.nan)
        monthly_profile = g.groupby(g["month_start"].dt.month)["units_sold"].mean()
        total_units = units.sum()
        rows.append({
            "region_id": r, "product_id": p,
            "avg_monthly_units": units.mean(),
            "cv_units": units.std() / units.mean() if units.mean() > 0 else 0.0,
            "yoy_growth_pct": (cur - prev) / prev * 100 if prev and prev > 0 else 0.0,
            "avg_discount_pct": g["discount_inr"].sum() / g["gross_revenue_inr"].sum() * 100,
            "spend_per_unit_inr": g["marketing_spend_inr"].sum() / total_units if total_units > 0 else 0.0,
            "seasonality_amplitude": (monthly_profile.max() - monthly_profile.min()) / monthly_profile.mean(),
            "campaign_frequency": g["campaign_flag"].mean(),
            "months_used": len(g),
            "history_end": g["month_start"].max(),
        })
    return pd.DataFrame(rows)


def evaluate_k(X: np.ndarray, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Silhouette, inertia, min cluster size, and seed stability (mean ARI) for each K."""
    rows = []
    for k in K_RANGE:
        base = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
        aris = [adjusted_rand_score(base.labels_, KMeans(n_clusters=k, n_init=20, random_state=s).fit(X).labels_)
                for s in STABILITY_SEEDS]
        rows.append({"k": k, "silhouette": float(silhouette_score(X, base.labels_)),
                     "inertia": float(base.inertia_), "min_cluster_size": int(np.bincount(base.labels_).min()),
                     "mean_ari_across_seeds": float(np.mean(aris))})
    return pd.DataFrame(rows)


def choose_k(k_table: pd.DataFrame, min_size: int = 2, tie_tol: float = 0.01) -> int:
    """Smallest K whose silhouette is within `tie_tol` of the best, among K with min cluster size >= min_size.

    With only 18 cells, a more complex segmentation must earn a clearly higher silhouette.
    """
    ok = k_table[k_table["min_cluster_size"] >= min_size]
    ok = ok if len(ok) else k_table
    best = ok["silhouette"].max()
    return int(ok[ok["silhouette"] >= best - tie_tol]["k"].min())


def label_clusters(centers_z: pd.DataFrame) -> dict[int, str]:
    """Descriptive names from standardized centroids (volume and volatility); never ROI claims."""
    labels = {}
    for c, row in centers_z.iterrows():
        volume = "high volume" if row["avg_monthly_units"] > 0.5 else ("low volume" if row["avg_monthly_units"] < -0.5 else "mid volume")
        vol = "volatile" if row["cv_units"] > 0.5 else ("stable" if row["cv_units"] < -0.5 else "moderate variability")
        growth = ", faster growth" if row["yoy_growth_pct"] > 0.5 else (", slower growth" if row["yoy_growth_pct"] < -0.5 else "")
        labels[int(c)] = f"{volume}, {vol}{growth}"
    seen: dict[str, int] = {}
    for c in sorted(labels):
        seen[labels[c]] = seen.get(labels[c], 0) + 1
        if list(labels.values()).count(labels[c]) > 1:
            labels[c] = f"{labels[c]} (#{seen[labels[c]]})"
    return labels


def run_segmentation(panel: pd.DataFrame, seed: int = config.RANDOM_SEED) -> dict[str, pd.DataFrame]:
    """Full pipeline; returns segments, profiles, k diagnostics and the raw descriptors."""
    feats = market_cell_features(panel)
    scaler = StandardScaler().fit(feats[SEGMENT_FEATURES])
    X = scaler.transform(feats[SEGMENT_FEATURES])
    k_table = evaluate_k(X, seed)
    k = choose_k(k_table)
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)

    centers_z = pd.DataFrame(km.cluster_centers_, columns=SEGMENT_FEATURES)
    names = label_clusters(centers_z)
    segments = feats.copy()
    segments["cluster"] = km.labels_
    segments["segment_label"] = segments["cluster"].map(names)
    segments["region_name"] = segments["region_id"].map(config.REGIONS)
    segments["product_name"] = segments["product_id"].map(config.PRODUCTS)
    pcs = PCA(n_components=2, random_state=seed).fit(X)
    segments[["pc1", "pc2"]] = pcs.transform(X)

    profiles = segments.groupby(["cluster", "segment_label"])[SEGMENT_FEATURES].mean().reset_index()
    profiles["n_cells"] = segments.groupby("cluster").size().to_numpy()
    centers_z.insert(0, "cluster", range(k))
    centers_z.insert(1, "segment_label", centers_z["cluster"].map(names))
    k_table["chosen"] = k_table["k"] == k
    k_table["pca_explained_variance_2d"] = float(pcs.explained_variance_ratio_.sum())
    return {"segments": segments, "profiles": profiles, "centers_z": centers_z, "k_table": k_table}


def save_artifacts(result: dict[str, pd.DataFrame]) -> None:
    """CSV outputs and the PCA / centroid heatmap figures."""
    result["segments"].to_csv(config.CLUSTERS_DIR / "market_segments.csv", index=False)
    result["profiles"].to_csv(config.CLUSTERS_DIR / "cluster_profiles.csv", index=False)
    result["centers_z"].to_csv(config.CLUSTERS_DIR / "cluster_centers_standardized.csv", index=False)
    result["k_table"].to_csv(config.CLUSTERS_DIR / "k_selection.csv", index=False)

    seg = result["segments"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, g in seg.groupby("segment_label"):
        ax.scatter(g["pc1"], g["pc2"], s=80, label=label)
        for _, r in g.iterrows():
            ax.annotate(f"{r['region_name'][:5]}-{r['product_id']}", (r["pc1"], r["pc2"]), fontsize=7)
    ax.set_xlabel("PC1 (standardized descriptors)")
    ax.set_ylabel("PC2")
    ax.set_title("Market cells in PCA space (visual aid only; synthetic, train period <= 2023)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "cluster_pca.png", dpi=120)
    plt.close(fig)

    cz = result["centers_z"].set_index("segment_label")[SEGMENT_FEATURES]
    fig, ax = plt.subplots(figsize=(9, 3 + 0.5 * len(cz)))
    im = ax.imshow(cz.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(SEGMENT_FEATURES)), SEGMENT_FEATURES, rotation=35, ha="right")
    ax.set_yticks(range(len(cz)), cz.index)
    fig.colorbar(im, ax=ax, label="standardized centroid (z)")
    ax.set_title("Cluster centroids (z-scores)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "cluster_centroids.png", dpi=120)
    plt.close(fig)
