"""Stage 1: unsupervised lesion segmentation with a Gaussian Mixture Model.

A single global GMM is fit on pooled HSV pixels sampled across the training
images. Each Gaussian component is then mapped to a semantic class (rust /
green / background) based on its mean hue and saturation. At inference time,
every pixel is assigned to its most likely component, giving rust / green /
leaf / background masks for an image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import joblib
import numpy as np
from sklearn.mixture import GaussianMixture

from .io import leaf_mask as compute_leaf_mask
from .io import to_hsv

RUST = "rust"
GREEN = "green"
BACKGROUND = "background"
SEMANTIC_CLASSES = (RUST, GREEN, BACKGROUND)


def classify_component(hue: float, sat: float, val: float) -> str:
    """Map a single HSV mean to a semantic class.

    Hue is on OpenCV's 0-179 scale. Rust/brown/orange lesions live at the warm
    (red->orange->yellow) end which wraps around 0; healthy tissue is green.
    Low-saturation or very dark/bright means are treated as background.
    """
    if sat < 40 or val < 40 or val > 245:
        return BACKGROUND
    if 35 <= hue <= 90:
        return GREEN
    # Warm colors: red (~0) through orange/brown (~30), plus the red wrap (>=160).
    if hue < 35 or hue >= 160:
        return RUST
    return BACKGROUND


class LeafGMM:
    """A fitted GMM plus its component -> semantic-class mapping."""

    def __init__(self, gmm: GaussianMixture, label_map: Dict[int, str]):
        self.gmm = gmm
        self.label_map = dict(label_map)

    @property
    def n_components(self) -> int:
        return self.gmm.n_components

    def components_for(self, semantic: str) -> List[int]:
        return [c for c, lab in self.label_map.items() if lab == semantic]

    def segment(self, bgr: np.ndarray, use_leaf_mask: bool = True) -> Dict[str, np.ndarray]:
        """Segment an image into boolean masks.

        Returns a dict with keys ``components`` (int label per pixel), ``rust``,
        ``green``, ``background`` and ``leaf`` (rust | green within foreground).
        """
        hsv = to_hsv(bgr)
        h, w = hsv.shape[:2]
        flat = hsv.reshape(-1, 3).astype(np.float64)
        comp = self.gmm.predict(flat).reshape(h, w)

        # Vectorized component -> semantic mapping.
        sem_lookup = np.array(
            [SEMANTIC_CLASSES.index(self.label_map[c]) for c in range(self.n_components)]
        )
        sem = sem_lookup[comp]

        rust = sem == SEMANTIC_CLASSES.index(RUST)
        green = sem == SEMANTIC_CLASSES.index(GREEN)
        background = sem == SEMANTIC_CLASSES.index(BACKGROUND)

        if use_leaf_mask:
            fg = compute_leaf_mask(hsv).astype(bool)
            rust = rust & fg
            green = green & fg
            background = background | (~fg)

        leaf = rust | green
        return {
            "components": comp,
            "rust": rust,
            "green": green,
            "background": background,
            "leaf": leaf,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"gmm": self.gmm, "label_map": self.label_map}, path)

    @classmethod
    def load(cls, path: str | Path) -> "LeafGMM":
        obj = joblib.load(path)
        return cls(obj["gmm"], obj["label_map"])


def _sample_pixels(
    images: Iterable[np.ndarray],
    max_pixels_per_image: int,
    use_leaf_mask: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    samples: List[np.ndarray] = []
    for bgr in images:
        hsv = to_hsv(bgr)
        px = hsv.reshape(-1, 3).astype(np.float64)
        if use_leaf_mask:
            m = compute_leaf_mask(hsv).reshape(-1).astype(bool)
            if m.any():
                px = px[m]
        if px.shape[0] == 0:
            continue
        if px.shape[0] > max_pixels_per_image:
            idx = rng.choice(px.shape[0], size=max_pixels_per_image, replace=False)
            px = px[idx]
        samples.append(px)
    if not samples:
        raise ValueError("No pixels sampled; check that images were provided.")
    return np.vstack(samples)


def build_label_map(gmm: GaussianMixture) -> Dict[int, str]:
    """Assign each component to a semantic class from its mean HSV."""
    label_map: Dict[int, str] = {}
    for i, (h, s, v) in enumerate(gmm.means_):
        label_map[i] = classify_component(float(h), float(s), float(v))
    return label_map


def train_gmm(
    images: Iterable[np.ndarray],
    k: int = 4,
    max_pixels_per_image: int = 20000,
    use_leaf_mask: bool = True,
    covariance_type: str = "full",
    random_state: int = 0,
) -> LeafGMM:
    """Fit a GMM on pooled HSV pixels and map components to semantic classes."""
    rng = np.random.default_rng(random_state)
    images = list(images)
    X = _sample_pixels(images, max_pixels_per_image, use_leaf_mask, rng)
    gmm = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        random_state=random_state,
        reg_covar=1e-4,
    )
    gmm.fit(X)
    label_map = build_label_map(gmm)
    return LeafGMM(gmm, label_map)
