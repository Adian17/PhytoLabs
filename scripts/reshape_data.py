"""Reshape raw class folders into the train/val layout the code expects.

Example
-------
    python -m scripts.reshape_data \
        --healthy-src data/raw/wheat/Healthy \
        --rust-src    data/raw/wheat/leaf_rust \
        --out data --val-fraction 0.2

Produces:
    data/train/healthy/*  data/train/rust/*
    data/val/healthy/*    data/val/rust/*
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _images(folder: Path):
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]


def _split_class(src: Path, out: Path, cls: str, val_fraction: float, seed: int, copy: bool) -> None:
    files = _images(src)
    if not files:
        raise FileNotFoundError(f"No images found in {src}")
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = max(1, int(round(len(files) * val_fraction)))
    val_files = set(files[:n_val])

    for split in ("train", "val"):
        (out / split / cls).mkdir(parents=True, exist_ok=True)

    for f in files:
        split = "val" if f in val_files else "train"
        dst = out / split / cls / f.name
        if copy:
            shutil.copy2(f, dst)
        else:
            shutil.move(str(f), str(dst))
    print(f"[{cls}] {len(files)} files -> {len(files) - n_val} train / {n_val} val")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--healthy-src", required=True, type=Path)
    ap.add_argument("--rust-src", required=True, type=Path)
    ap.add_argument("--out", default=Path("data"), type=Path)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--move", action="store_true", help="Move files instead of copying")
    args = ap.parse_args()

    copy = not args.move
    _split_class(args.healthy_src, args.out, "healthy", args.val_fraction, args.seed, copy)
    _split_class(args.rust_src, args.out, "rust", args.val_fraction, args.seed, copy)
    print(f"Done. Dataset ready at {args.out}/")


if __name__ == "__main__":
    main()
