#!/usr/bin/env bash
#
# Download a wheat-rust dataset from Kaggle and reshape it into the layout the
# PhytoLabs code expects:
#
#   data/
#     train/  healthy/*.jpg   rust/*.jpg
#     val/    healthy/*.jpg   rust/*.jpg
#
# Prerequisites
# -------------
#   1. pip install kaggle
#   2. Create a Kaggle API token (Account -> Settings -> "Create New Token"),
#      which downloads kaggle.json, then:
#        mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
#        chmod 600 ~/.kaggle/kaggle.json
#
# This script CANNOT run without your own Kaggle credentials. It documents the
# exact steps; review and run section-by-section as needed.

set -euo pipefail

DATA_DIR="${1:-data}"
RAW_DIR="${DATA_DIR}/raw"
VAL_FRACTION="${VAL_FRACTION:-0.2}"

mkdir -p "${RAW_DIR}"

# ---------------------------------------------------------------------------
# Option A (recommended for color segmentation): single-leaf, close-view set.
# "Wheat Leaf Dataset" has classes: Healthy, septoria, stripe rust (varies by
# mirror). For brown rust use the leaf/brown-rust class as "rust".
# ---------------------------------------------------------------------------
# kaggle datasets download -d olyadgetch/wheat-leaf-dataset -p "${RAW_DIR}" --unzip

# ---------------------------------------------------------------------------
# Option B (alternative): CGIAR Computer Vision for Crop Disease.
# Multiple leaves per image + complex backgrounds (harder for HSV/GMM).
# Classes include leaf_rust (= brown rust), stem_rust, healthy_wheat.
# ---------------------------------------------------------------------------
# kaggle datasets download -d shadabhussain/cgiar-computer-vision-for-crop-disease -p "${RAW_DIR}" --unzip

cat <<'EOF'
-----------------------------------------------------------------------------
NEXT STEPS (manual, depends on which dataset you downloaded):

1. Uncomment ONE of the `kaggle datasets download` lines above and run this
   script again, OR download manually from kaggle.com and unzip into data/raw/.

2. Identify the two folders you care about for SPRING WHEAT BROWN RUST:
     - a "healthy" class folder
     - a "leaf rust" / "brown rust" class folder
   (Ignore stem rust, yellow/stripe rust, powdery mildew, scab, etc.)

3. Reshape into the expected layout. Use the helper:
     python -m scripts.reshape_data \
        --healthy-src data/raw/<...>/Healthy \
        --rust-src    data/raw/<...>/leaf_rust \
        --out data --val-fraction 0.2

   (Or just copy files by hand into data/{train,val}/{healthy,rust}/.)

4. Sanity check the pipeline first WITHOUT real data:
     python -m phytolabs.cli make-synthetic --data-dir data
-----------------------------------------------------------------------------
EOF
