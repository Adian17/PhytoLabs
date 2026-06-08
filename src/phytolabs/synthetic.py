"""Synthetic leaf images so the pipeline runs end-to-end before the real dataset.

These are crude on purpose: a green elliptical "leaf" on a near-white
background, with rust images getting orange/brown speckles. Good enough to
exercise the GMM, feature extraction, and classifier wiring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def _draw_leaf(rng: np.random.Generator, size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Return (bgr image, leaf mask) of a green leaf on a light background."""
    img = np.full((size, size, 3), 245, dtype=np.uint8)  # near-white background
    img += rng.integers(-5, 6, img.shape, dtype=np.int16).astype(np.int16).clip(-5, 5).astype(np.uint8)

    center = (size // 2 + int(rng.integers(-15, 16)), size // 2 + int(rng.integers(-15, 16)))
    axes = (int(size * 0.32), int(size * 0.42))
    angle = int(rng.integers(0, 180))

    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)

    # Healthy green tissue with slight texture.
    green = np.zeros_like(img)
    base = np.array([60, 150, 70], dtype=np.int16)  # BGR-ish green
    noise = rng.integers(-20, 21, (size, size, 3))
    green_tex = np.clip(base + noise, 0, 255).astype(np.uint8)
    img[mask > 0] = green_tex[mask > 0]
    return img, mask


def make_leaf_image(
    rng: np.random.Generator,
    diseased: bool,
    size: int = 256,
) -> np.ndarray:
    """Create one synthetic leaf image (BGR)."""
    img, mask = _draw_leaf(rng, size)
    if diseased:
        n_spots = int(rng.integers(15, 45))
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            for _ in range(n_spots):
                k = int(rng.integers(0, len(xs)))
                cx, cy = int(xs[k]), int(ys[k])
                r = int(rng.integers(2, 7))
                # Rust/brown lesion color in BGR (orange-brown).
                color = (
                    int(rng.integers(10, 50)),
                    int(rng.integers(60, 110)),
                    int(rng.integers(150, 210)),
                )
                cv2.circle(img, (cx, cy), r, color, -1)
    return img


def make_synthetic_dataset(
    root: str | Path,
    n_per_class: int = 12,
    val_fraction: float = 0.34,
    size: int = 256,
    seed: int = 0,
    overwrite: bool = False,
) -> Path:
    """Write a synthetic dataset under ``root`` matching the expected layout.

    Creates ``root/train/{healthy,rust}`` and ``root/val/{healthy,rust}``.
    """
    root = Path(root)
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(n_per_class * val_fraction)))
    n_train = max(1, n_per_class - n_val)

    for split, count in (("train", n_train), ("val", n_val)):
        for cls, diseased in (("healthy", False), ("rust", True)):
            folder = root / split / cls
            folder.mkdir(parents=True, exist_ok=True)
            if overwrite:
                for f in folder.glob("*.png"):
                    f.unlink()
            for i in range(count):
                img = make_leaf_image(rng, diseased=diseased, size=size)
                cv2.imwrite(str(folder / f"{cls}_{i:03d}.png"), img)
    return root
