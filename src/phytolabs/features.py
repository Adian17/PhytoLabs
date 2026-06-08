"""Stage 1 -> Stage 2 bridge: per-image features from segmentation masks.

Given a rust (lesion) mask and a leaf (foreground) mask, compute the
quantitative, image-level features that feed the classifier.
"""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

FEATURE_NAMES: List[str] = [
    "lesion_area_fraction",
    "blob_count",
    "blob_size_mean",
    "blob_size_std",
    "blob_size_max",
    "blob_density",
]


def extract_features(
    rust_mask: np.ndarray,
    leaf_mask: np.ndarray | None = None,
    min_blob_area: int = 5,
) -> Dict[str, float]:
    """Compute lesion features from boolean masks.

    Parameters
    ----------
    rust_mask : boolean array of lesion pixels.
    leaf_mask : boolean array of foreground (leaf) pixels. If ``None``, the full
        image is used as the denominator.
    min_blob_area : connected components smaller than this are ignored as noise.
    """
    rust = rust_mask.astype(np.uint8)

    if leaf_mask is not None:
        leaf_area = int(leaf_mask.astype(bool).sum())
    else:
        leaf_area = rust.size
    leaf_area = max(leaf_area, 1)

    rust_area = int(rust.sum())
    lesion_area_fraction = rust_area / leaf_area

    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(rust, connectivity=8)
    # Label 0 is the background of the mask; skip it.
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64) if n_labels > 1 else np.array([])
    areas = areas[areas >= min_blob_area]

    blob_count = int(areas.size)
    if blob_count > 0:
        blob_size_mean = float(areas.mean())
        blob_size_std = float(areas.std())
        blob_size_max = float(areas.max())
    else:
        blob_size_mean = blob_size_std = blob_size_max = 0.0

    # Blobs per 10k leaf pixels keeps this scale-invariant across image sizes.
    blob_density = blob_count / leaf_area * 1e4

    return {
        "lesion_area_fraction": lesion_area_fraction,
        "blob_count": float(blob_count),
        "blob_size_mean": blob_size_mean,
        "blob_size_std": blob_size_std,
        "blob_size_max": blob_size_max,
        "blob_density": blob_density,
    }


def features_to_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert a feature dict to a fixed-order vector matching FEATURE_NAMES."""
    return np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)
