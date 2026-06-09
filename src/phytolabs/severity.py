"""Rust-severity grading derived from Stage-1 segmentation.

Severity here is the **percent of leaf area covered by rust pustules** — the
standard "percent leaf area affected" used in wheat-rust scoring (cf. the
modified Cobb scale). It is computed directly from the segmentation masks and is
therefore independent of the logistic-regression decision: even when the binary
classifier calls an image healthy, a small but non-zero severity flags a
trace-level infection.

There is **no severity ground truth** in the datasets (labels are only
healthy/diseased), so the grade is a deterministic, interpretable readout — not
a learned, accuracy-validated quantity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

# Ordinal grades, from least to most severe.
SEVERITY_GRADES: Tuple[str, ...] = ("none", "mild", "moderate", "severe")

# Default cut points (in percent of leaf area) between consecutive grades:
#   none     : pct < 1
#   mild     : 1  <= pct < 10
#   moderate : 10 <= pct < 25
#   severe   : pct >= 25
# Tunable; see the threshold-suggestion cell in the Colab notebook.
DEFAULT_THRESHOLDS: Tuple[float, ...] = (1.0, 10.0, 25.0)


@dataclass(frozen=True)
class SeverityResult:
    """The severity readout for one image."""

    percent: float  # percent of leaf area covered by rust (0-100)
    grade: str  # one of SEVERITY_GRADES
    index: int  # ordinal index into SEVERITY_GRADES (0 = none)


def severity_percent(rust_mask: np.ndarray, leaf_mask: np.ndarray | None = None) -> float:
    """Percent of leaf area covered by rust pixels (0-100).

    Mirrors ``features.lesion_area_fraction`` but expressed as a percentage. If
    ``leaf_mask`` is ``None`` the whole image is used as the denominator.
    """
    rust_area = int(np.asarray(rust_mask).astype(bool).sum())
    if leaf_mask is not None:
        leaf_area = int(np.asarray(leaf_mask).astype(bool).sum())
    else:
        leaf_area = int(np.asarray(rust_mask).size)
    leaf_area = max(leaf_area, 1)
    return 100.0 * rust_area / leaf_area


def grade_from_percent(
    pct: float, thresholds: Sequence[float] = DEFAULT_THRESHOLDS
) -> Tuple[str, int]:
    """Map a severity percent to an ordinal (grade, index).

    ``thresholds`` is the ascending list of cut points between grades; with the
    default ``(1, 10, 25)`` there are four grades. ``index`` counts how many cut
    points ``pct`` meets or exceeds.
    """
    thresholds = list(thresholds)
    if len(thresholds) != len(SEVERITY_GRADES) - 1:
        raise ValueError(
            f"Expected {len(SEVERITY_GRADES) - 1} thresholds for "
            f"{len(SEVERITY_GRADES)} grades, got {len(thresholds)}."
        )
    index = int(np.searchsorted(thresholds, pct, side="right"))
    return SEVERITY_GRADES[index], index


def grade_severity(
    rust_mask: np.ndarray,
    leaf_mask: np.ndarray | None = None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> SeverityResult:
    """Compute a full ``SeverityResult`` from segmentation masks."""
    pct = severity_percent(rust_mask, leaf_mask)
    grade, index = grade_from_percent(pct, thresholds)
    return SeverityResult(percent=pct, grade=grade, index=index)


def grade_features(
    features: dict, thresholds: Sequence[float] = DEFAULT_THRESHOLDS
) -> SeverityResult:
    """Compute severity from a feature dict that has ``lesion_area_fraction``."""
    pct = 100.0 * float(features["lesion_area_fraction"])
    grade, index = grade_from_percent(pct, thresholds)
    return SeverityResult(percent=pct, grade=grade, index=index)
