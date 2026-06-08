"""Shared notebook setup: make `phytolabs` importable and locate data.

Used by the notebooks via `from _setup import DATA_DIR, ensure_dataset`.
Works whether or not the package is pip-installed (`pip install -e .`).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def has_real_data(data_dir: Path = DATA_DIR) -> bool:
    train = data_dir / "train"
    if not train.exists():
        return False
    for cls in ("healthy", "rust"):
        folder = train / cls
        if not folder.exists() or not any(folder.iterdir()):
            return False
    return True


def ensure_dataset(data_dir: Path = DATA_DIR, n_per_class: int = 16) -> Path:
    """Return a usable dataset dir, generating synthetic data if none exists."""
    if has_real_data(data_dir):
        print(f"Using dataset at {data_dir}")
        return data_dir
    from phytolabs.synthetic import make_synthetic_dataset

    print(f"No real dataset found; generating synthetic data at {data_dir}")
    make_synthetic_dataset(data_dir, n_per_class=n_per_class, overwrite=True)
    return data_dir
