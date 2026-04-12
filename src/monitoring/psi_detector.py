from __future__ import annotations

import numpy as np
import pandas as pd


def compute_psi(expected, actual, buckets: int = 10) -> float:
    expected_series = pd.Series(expected).dropna()
    actual_series = pd.Series(actual).dropna()

    if expected_series.empty or actual_series.empty:
        return 0.0

    epsilon = 1e-6

    if (
        pd.api.types.is_numeric_dtype(expected_series)
        and pd.api.types.is_numeric_dtype(actual_series)
    ):
        percentiles = np.linspace(0, 100, buckets + 1)
        bin_edges = np.unique(np.percentile(expected_series.astype(float), percentiles))
        if bin_edges.size < 2:
            return 0.0

        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        expected_counts, _ = np.histogram(expected_series.astype(float), bins=bin_edges)
        actual_counts, _ = np.histogram(actual_series.astype(float), bins=bin_edges)
    else:
        categories = sorted(
            set(expected_series.astype(str).unique()).union(actual_series.astype(str).unique())
        )
        expected_counts = (
            expected_series.astype(str).value_counts().reindex(categories, fill_value=0).to_numpy()
        )
        actual_counts = (
            actual_series.astype(str).value_counts().reindex(categories, fill_value=0).to_numpy()
        )

    expected_pct = np.clip(expected_counts / max(expected_counts.sum(), 1), epsilon, None)
    actual_pct = np.clip(actual_counts / max(actual_counts.sum(), 1), epsilon, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def detect_psi_drift(
    expected,
    actual,
    *,
    buckets: int = 10,
    threshold: float = 0.2,
) -> dict[str, float | bool]:
    psi_score = compute_psi(expected, actual, buckets=buckets)
    return {
        "drift_score": float(psi_score),
        "threshold": float(threshold),
        "detected": bool(psi_score > threshold),
    }
