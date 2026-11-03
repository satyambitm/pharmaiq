"""Stored K-Means market-cell segments (fit offline by `ml/train_kmeans.py`)."""

from __future__ import annotations

from ml.model_loader import ModelBundle

NOTES = [
    "Each cell is one region x product series described by history up to the cutoff month.",
    "K is chosen by silhouette; scores within 0.01 prefer the smaller K.",
    "On this synthetic data the three segments line up with the three products, so they mostly "
    "restate product differences rather than reveal hidden regional groups.",
]


def segments(bundle: ModelBundle) -> dict:
    ks = bundle.k_selection
    return {
        "k": int(ks.loc[ks["chosen"], "k"].iloc[0]),
        "features": list(bundle.kmeans["features"]),
        "history_cutoff": str(bundle.kmeans["cutoff"])[:10],
        "notes": NOTES,
        "cells": bundle.segments.to_dict("records"),
        "profiles": bundle.segment_profiles.to_dict("records"),
        "k_selection": ks.to_dict("records"),
    }
