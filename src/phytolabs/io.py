"""Image I/O and color-space helpers.

OpenCV is used only for decoding images, resizing, and the BGR->HSV conversion.
Everything downstream operates on plain NumPy arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Tuple

import cv2
import numpy as np

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def resize_max(bgr: np.ndarray, max_size: int) -> np.ndarray:
    """Downscale so the longest side is at most ``max_size`` (never upscales)."""
    if max_size is None:
        return bgr
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_size:
        return bgr
    scale = max_size / float(longest)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(bgr, new_size, interpolation=cv2.INTER_AREA)


def load_image(path: str | Path, max_size: int | None = 512) -> np.ndarray:
    """Load an image as a BGR uint8 array, optionally downscaled."""
    path = Path(path)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return resize_max(bgr, max_size)


def to_hsv(bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR uint8 image to HSV (H in [0,179], S/V in [0,255])."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def leaf_mask(hsv: np.ndarray, sat_min: int = 30, val_min: int = 25, val_max: int = 250) -> np.ndarray:
    """Heuristic foreground (leaf) mask.

    Backgrounds in close-view leaf photos are usually near-white (low saturation)
    or near-black. The leaf itself is colored (higher saturation). This is a
    coarse pre-mask; the GMM still does the rust/green separation.
    """
    h, s, v = cv2.split(hsv)
    mask = (s >= sat_min) & (v >= val_min) & (v <= val_max)
    mask = mask.astype(np.uint8)
    # Clean up speckle with a small morphological open/close.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def iter_image_paths(folder: str | Path) -> Iterator[Path]:
    """Yield image file paths in a folder, sorted, filtered by extension."""
    folder = Path(folder)
    if not folder.exists():
        return
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p


def load_folder(folder: str | Path, max_size: int | None = 512, limit: int | None = None) -> Tuple[List[np.ndarray], List[Path]]:
    """Load all images in a folder. Returns (images, paths)."""
    images: List[np.ndarray] = []
    paths: List[Path] = []
    for i, p in enumerate(iter_image_paths(folder)):
        if limit is not None and i >= limit:
            break
        images.append(load_image(p, max_size=max_size))
        paths.append(p)
    return images, paths
