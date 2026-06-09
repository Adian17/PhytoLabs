"""Command-line interface for the PhytoLabs pipeline.

Subcommands:
    make-synthetic   Generate a synthetic dataset (works with no real data).
    train-gmm        Fit the Stage 1 GMM and save it.
    build-features   Turn images into a feature table (npz) using the GMM.
    train-logreg     Train the from-scratch logistic regression + evaluate.
    predict          Run the full pipeline on a single image.

Artifacts are written to (by default) the ``artifacts/`` directory:
    gmm.joblib, features_train.npz, features_val.npz, logreg.joblib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import pipeline
from .calibration import band_summary, expected_calibration_error
from .features import FEATURE_NAMES
from .logreg import LogisticRegressionSGD
from .metrics import classification_report
from .segmentation import LeafGMM
from .synthetic import make_synthetic_dataset


def _save_feature_table(path: Path, X: np.ndarray, y: np.ndarray, paths) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, y=y, paths=np.array(paths, dtype=object))


def _load_feature_table(path: Path):
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"], list(data["paths"])


def cmd_make_synthetic(args: argparse.Namespace) -> None:
    root = make_synthetic_dataset(
        args.data_dir, n_per_class=args.n_per_class, size=args.size, seed=args.seed, overwrite=True
    )
    print(f"Synthetic dataset written to {root}/ (train/ and val/, classes: healthy, rust)")


def cmd_train_gmm(args: argparse.Namespace) -> None:
    train_dir = Path(args.data_dir) / "train"
    gmm = pipeline.fit_gmm_from_dir(
        train_dir, k=args.k, max_size=args.max_size, use_leaf_mask=not args.no_leaf_mask
    )
    out = Path(args.gmm)
    gmm.save(out)
    print(f"Trained GMM (k={args.k}) saved to {out}")
    print("Component -> semantic label map:")
    for c in range(gmm.n_components):
        h, s, v = gmm.gmm.means_[c]
        print(f"  component {c}: HSV mean=({h:.0f},{s:.0f},{v:.0f}) -> {gmm.label_map[c]}")


def cmd_build_features(args: argparse.Namespace) -> None:
    gmm = LeafGMM.load(args.gmm)
    for split in args.splits:
        split_dir = Path(args.data_dir) / split
        X, y, paths = pipeline.build_feature_table(
            split_dir, gmm, max_size=args.max_size, use_leaf_mask=not args.no_leaf_mask
        )
        out = Path(args.artifacts) / f"features_{split}.npz"
        _save_feature_table(out, X, y, paths)
        print(f"[{split}] {X.shape[0]} images, {X.shape[1]} features -> {out}")


def cmd_train_logreg(args: argparse.Namespace) -> None:
    X_train, y_train, _ = _load_feature_table(Path(args.artifacts) / "features_train.npz")
    model = LogisticRegressionSGD(
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, l2=args.l2, random_state=args.seed
    )
    model.fit(X_train, y_train)
    out = Path(args.logreg)
    model.save(out)
    print(f"Trained logistic regression saved to {out}")
    print("Learned weights (standardized features):")
    for name, w in zip(FEATURE_NAMES, model.w):
        print(f"  {name}: {w:+.3f}")
    print(f"  bias: {model.b:+.3f}")

    train_report = classification_report(
        y_train, model.predict(X_train), model.predict_proba(X_train)
    )
    print("\nTrain metrics:", json.dumps(train_report, indent=2))

    val_path = Path(args.artifacts) / "features_val.npz"
    if val_path.exists():
        X_val, y_val, _ = _load_feature_table(val_path)
        proba = model.predict_proba(X_val)
        report = classification_report(y_val, model.predict(X_val), proba)
        print("\nValidation metrics:", json.dumps(report, indent=2))
        print("Band summary:", band_summary(proba, args.band_low, args.band_high))
        print(f"Expected calibration error: {expected_calibration_error(y_val, proba):.4f}")


def cmd_predict(args: argparse.Namespace) -> None:
    gmm = LeafGMM.load(args.gmm)
    model = LogisticRegressionSGD.load(args.logreg)
    result, seg, bgr = pipeline.predict_image(
        args.image, gmm, model, band=(args.band_low, args.band_high), max_size=args.max_size
    )
    print(f"Image: {args.image}")
    print(f"  P(rust) = {result['probability']:.3f}  ->  {result['label']}")
    print(f"  Severity = {result['severity']} ({result['severity_percent']:.1f}% leaf area)")
    print("  Features:")
    for name in FEATURE_NAMES:
        print(f"    {name}: {result['features'][name]:.4f}")

    if args.save_overlay:
        import cv2
        from .viz import overlay_lesions

        overlay = overlay_lesions(bgr, seg["rust"])
        out = Path(args.save_overlay)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), overlay)
        print(f"  Lesion overlay saved to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phytolabs", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common_data = {"default": "data", "help": "Dataset root (contains train/ and val/)"}
    common_gmm = {"default": "artifacts/gmm.joblib", "help": "Path to GMM artifact"}
    common_logreg = {"default": "artifacts/logreg.joblib", "help": "Path to logreg artifact"}
    common_artifacts = {"default": "artifacts", "help": "Artifacts directory"}

    p = sub.add_parser("make-synthetic", help="Generate a synthetic dataset")
    p.add_argument("--data-dir", **common_data)
    p.add_argument("--n-per-class", type=int, default=16)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_make_synthetic)

    p = sub.add_parser("train-gmm", help="Fit the Stage 1 GMM")
    p.add_argument("--data-dir", **common_data)
    p.add_argument("--gmm", **common_gmm)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--max-size", type=int, default=512)
    p.add_argument("--no-leaf-mask", action="store_true")
    p.set_defaults(func=cmd_train_gmm)

    p = sub.add_parser("build-features", help="Build feature tables from images")
    p.add_argument("--data-dir", **common_data)
    p.add_argument("--gmm", **common_gmm)
    p.add_argument("--artifacts", **common_artifacts)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--max-size", type=int, default=512)
    p.add_argument("--no-leaf-mask", action="store_true")
    p.set_defaults(func=cmd_build_features)

    p = sub.add_parser("train-logreg", help="Train the from-scratch logistic regression")
    p.add_argument("--artifacts", **common_artifacts)
    p.add_argument("--logreg", **common_logreg)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--band-low", type=float, default=0.45)
    p.add_argument("--band-high", type=float, default=0.55)
    p.set_defaults(func=cmd_train_logreg)

    p = sub.add_parser("predict", help="Predict on a single image")
    p.add_argument("image", help="Path to an image")
    p.add_argument("--gmm", **common_gmm)
    p.add_argument("--logreg", **common_logreg)
    p.add_argument("--band-low", type=float, default=0.45)
    p.add_argument("--band-high", type=float, default=0.55)
    p.add_argument("--max-size", type=int, default=512)
    p.add_argument("--save-overlay", default=None, help="Optional path to save the lesion overlay")
    p.set_defaults(func=cmd_predict)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
