"""Visualization helpers: lesion overlay (the product UX) + diagnostic plots.

Functions that draw plots import matplotlib lazily and return Figure objects so
they work both in notebooks and headless scripts.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from .calibration import reliability_curve
from .metrics import confusion_matrix


def overlay_lesions(
    bgr: np.ndarray,
    rust_mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.5,
    outline: bool = True,
) -> np.ndarray:
    """Highlight lesion pixels on the original image. Returns a BGR image.

    ``color`` is in BGR (default red). With ``outline`` true, lesion contours are
    drawn for a crisper look.
    """
    out = bgr.copy()
    mask = rust_mask.astype(bool)
    tint = np.zeros_like(bgr)
    tint[mask] = color
    out = cv2.addWeighted(tint, alpha, out, 1.0, 0.0)
    out[~mask] = bgr[~mask]
    if outline:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, color, 1)
    return out


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV) to RGB for matplotlib display."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def plot_segmentation(bgr: np.ndarray, seg: Dict[str, np.ndarray], title: str | None = None):
    """Show original, lesion overlay, and the rust/green/background masks."""
    import matplotlib.pyplot as plt

    overlay = overlay_lesions(bgr, seg["rust"])
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(bgr_to_rgb(bgr))
    axes[0].set_title("Original")
    axes[1].imshow(bgr_to_rgb(overlay))
    axes[1].set_title("Lesion overlay")
    axes[2].imshow(seg["rust"], cmap="Reds")
    axes[2].set_title("Rust mask")
    axes[3].imshow(seg["green"], cmap="Greens")
    axes[3].set_title("Green mask")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def overlay_gallery(samples: Sequence[Tuple[np.ndarray, np.ndarray, str]], ncols: int = 4):
    """Grid of lesion overlays. Each sample is (bgr, rust_mask, caption)."""
    import matplotlib.pyplot as plt

    n = len(samples)
    ncols = min(ncols, max(n, 1))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (bgr, mask, caption) in zip(axes, samples):
        ax.imshow(bgr_to_rgb(overlay_lesions(bgr, mask)))
        ax.set_title(caption, fontsize=9)
    fig.tight_layout()
    return fig


def plot_feature_histograms(X: np.ndarray, y: np.ndarray, feature_names: List[str]):
    """Per-feature histograms split by class (healthy vs rust)."""
    import matplotlib.pyplot as plt

    n_feat = X.shape[1]
    ncols = 3
    nrows = int(np.ceil(n_feat / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, name in enumerate(feature_names):
        ax = axes[i]
        ax.hist(X[y == 0, i], bins=20, alpha=0.6, label="healthy")
        ax.hist(X[y == 1, i], bins=20, alpha=0.6, label="rust")
        ax.set_title(name)
        ax.legend()
    for j in range(n_feat, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    return fig


def plot_loss(loss_history: Sequence[float]):
    """Training loss curve."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(loss_history)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (BCE + L2)")
    ax.set_title("Training loss")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels=("healthy", "rust")):
    """Annotated 2x2 confusion matrix."""
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_roc(y_true: np.ndarray, proba: np.ndarray):
    """ROC curve computed from probability thresholds."""
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).astype(int).ravel()
    proba = np.asarray(proba, dtype=np.float64).ravel()
    thresholds = np.unique(np.concatenate([[0.0], proba, [1.0]]))[::-1]
    tpr, fpr = [], []
    p = int(np.sum(y_true == 1))
    n = int(np.sum(y_true == 0))
    for t in thresholds:
        pred = proba >= t
        tp = int(np.sum(pred & (y_true == 1)))
        fp = int(np.sum(pred & (y_true == 0)))
        tpr.append(tp / p if p else 0.0)
        fpr.append(fp / n if n else 0.0)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, marker=".")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    fig.tight_layout()
    return fig


def plot_reliability(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10):
    """Reliability diagram (calibration curve)."""
    import matplotlib.pyplot as plt

    mean_pred, obs_frac, counts = reliability_curve(y_true, proba, n_bins)
    mask = counts > 0
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfectly calibrated")
    ax.plot(mean_pred[mask], obs_frac[mask], marker="o", label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title("Reliability diagram")
    ax.legend()
    fig.tight_layout()
    return fig
