"""PhytoLabs: wheat brown-rust detection.

Stage 1: GMM (HSV) unsupervised lesion segmentation + per-image features.
Stage 2: from-scratch logistic regression + SGD over those features, with a
"suspicious" confidence band.
"""

from . import io, segmentation, features, logreg, calibration, pipeline, viz, synthetic, severity

__all__ = [
    "io",
    "segmentation",
    "features",
    "logreg",
    "calibration",
    "pipeline",
    "viz",
    "synthetic",
    "severity",
]

__version__ = "0.1.0"
