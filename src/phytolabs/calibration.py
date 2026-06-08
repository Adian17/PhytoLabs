"""Confidence band and calibration helpers.

The logistic-regression output is already a probability. A configurable band
around 0.5 (default 0.45-0.55) carves out a third "suspicious" class for cases
the model is unsure about. The reliability curve lets you check how well the
predicted probabilities match observed frequencies.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

HEALTHY = "healthy"
DISEASED = "diseased"
SUSPICIOUS = "suspicious"


def classify_with_band(
    proba: np.ndarray,
    low: float = 0.45,
    high: float = 0.55,
) -> np.ndarray:
    """Map probabilities to {healthy, suspicious, diseased} using a band."""
    proba = np.asarray(proba, dtype=np.float64)
    labels = np.full(proba.shape, SUSPICIOUS, dtype=object)
    labels[proba >= high] = DISEASED
    labels[proba <= low] = HEALTHY
    return labels


def band_summary(proba: np.ndarray, low: float = 0.45, high: float = 0.55) -> Dict[str, int]:
    """Count how many predictions fall into each band class."""
    labels = classify_with_band(proba, low, high)
    return {
        HEALTHY: int(np.sum(labels == HEALTHY)),
        SUSPICIOUS: int(np.sum(labels == SUSPICIOUS)),
        DISEASED: int(np.sum(labels == DISEASED)),
    }


def reliability_curve(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a reliability (calibration) curve.

    Returns (mean_predicted, observed_fraction, bin_counts) over ``n_bins``
    equal-width probability bins. Empty bins are returned as NaN.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    proba = np.asarray(proba, dtype=np.float64).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_pred = np.full(n_bins, np.nan)
    obs_frac = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (proba >= lo) & (proba < hi) if i < n_bins - 1 else (proba >= lo) & (proba <= hi)
        counts[i] = int(in_bin.sum())
        if counts[i] > 0:
            mean_pred[i] = float(proba[in_bin].mean())
            obs_frac[i] = float(y_true[in_bin].mean())
    return mean_pred, obs_frac, counts


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: weighted gap between confidence and accuracy."""
    mean_pred, obs_frac, counts = reliability_curve(y_true, proba, n_bins)
    total = counts.sum()
    if total == 0:
        return float("nan")
    mask = counts > 0
    gaps = np.abs(mean_pred[mask] - obs_frac[mask])
    weights = counts[mask] / total
    return float(np.sum(weights * gaps))
