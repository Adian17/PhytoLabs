"""End-to-end glue: image -> segmentation -> features -> probability -> label."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .calibration import classify_with_band
from .features import extract_features, features_to_vector, FEATURE_NAMES
from .io import iter_image_paths, load_image
from .logreg import LogisticRegressionSGD
from .segmentation import LeafGMM, train_gmm

# Folder name -> binary label used across the project.
DEFAULT_CLASS_MAP: Dict[str, int] = {"healthy": 0, "rust": 1}


def image_to_features(
    bgr: np.ndarray,
    leaf_gmm: LeafGMM,
    use_leaf_mask: bool = True,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Segment one image and return (feature dict, segmentation masks)."""
    seg = leaf_gmm.segment(bgr, use_leaf_mask=use_leaf_mask)
    feats = extract_features(seg["rust"], seg["leaf"])
    return feats, seg


def fit_gmm_from_dir(
    data_dir: str | Path,
    class_map: Dict[str, int] | None = None,
    k: int = 4,
    max_size: int = 512,
    limit_per_class: int | None = None,
    **train_kwargs,
) -> LeafGMM:
    """Train a GMM from all images under ``data_dir/<class>/``."""
    class_map = class_map or DEFAULT_CLASS_MAP
    data_dir = Path(data_dir)
    images: List[np.ndarray] = []
    for cls in class_map:
        folder = data_dir / cls
        for i, p in enumerate(iter_image_paths(folder)):
            if limit_per_class is not None and i >= limit_per_class:
                break
            images.append(load_image(p, max_size=max_size))
    if not images:
        raise FileNotFoundError(
            f"No images found under {data_dir} for classes {list(class_map)}. "
            f"See scripts/download_data.sh to set up the dataset."
        )
    return train_gmm(images, k=k, **train_kwargs)


def build_feature_table(
    data_dir: str | Path,
    leaf_gmm: LeafGMM,
    class_map: Dict[str, int] | None = None,
    max_size: int = 512,
    use_leaf_mask: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build (X, y, paths) for every image under ``data_dir/<class>/``."""
    class_map = class_map or DEFAULT_CLASS_MAP
    data_dir = Path(data_dir)
    rows: List[np.ndarray] = []
    labels: List[int] = []
    paths: List[str] = []
    for cls, y in class_map.items():
        for p in iter_image_paths(data_dir / cls):
            bgr = load_image(p, max_size=max_size)
            feats, _ = image_to_features(bgr, leaf_gmm, use_leaf_mask=use_leaf_mask)
            rows.append(features_to_vector(feats))
            labels.append(y)
            paths.append(str(p))
    if not rows:
        raise FileNotFoundError(
            f"No images found under {data_dir} for classes {list(class_map)}."
        )
    return np.vstack(rows), np.array(labels, dtype=int), paths


def predict_image(
    path: str | Path,
    leaf_gmm: LeafGMM,
    model: LogisticRegressionSGD,
    band: Tuple[float, float] = (0.45, 0.55),
    max_size: int = 512,
    use_leaf_mask: bool = True,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray], np.ndarray]:
    """Run the full pipeline on one image.

    Returns (result, segmentation masks, original BGR image). ``result`` has
    keys: probability, label, features.
    """
    bgr = load_image(path, max_size=max_size)
    feats, seg = image_to_features(bgr, leaf_gmm, use_leaf_mask=use_leaf_mask)
    x = features_to_vector(feats)[None, :]
    proba = float(model.predict_proba(x)[0])
    label = str(classify_with_band(np.array([proba]), band[0], band[1])[0])
    result = {
        "probability": proba,
        "label": label,
        "features": feats,
    }
    return result, seg, bgr
