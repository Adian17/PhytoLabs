# PhytoLabs — Wheat Brown-Rust Detection

A two-stage, classical-ML pipeline for detecting **spring wheat brown rust**
(a.k.a. leaf rust, *Puccinia triticina*) in leaf photos.

- **Stage 1 — GMM + EM (unsupervised segmentation).** A Gaussian Mixture Model
  is fit in HSV color space to cluster leaf pixels, separating rust-colored
  lesions from healthy green tissue and background. This yields quantitative
  per-image features (lesion area fraction, blob count, size distribution) and,
  as a bonus, a visual lesion overlay — the core PhytoLabs UX.
- **Stage 2 — Logistic Regression + SGD (classification).** Those features feed
  a logistic regression trained **from scratch** with mini-batch SGD, producing
  a calibrated disease probability. A confidence band (default 0.45–0.55) carves
  out a third **"suspicious / uncertain"** class.

### Scope (chosen constraints)
- Spring wheat **brown rust only** (one disease).
- **Image-level** classification. We do **not** claim per-region accuracy — the
  datasets only provide image-level labels.
- Datasets: Kaggle / CGIAR wheat-rust sets (PlantVillage lacks wheat).

## Architecture

```
image ─▶ BGR→HSV ─▶ GMM clusters ─▶ rust/green/bg masks ─▶ features ─▶ logreg(SGD) ─▶ P(rust)
                                          │                                              │
                                          └▶ lesion overlay (UX)              0.45–0.55 → "suspicious"
```

| Module | Role |
| --- | --- |
| `phytolabs/io.py` | image load/resize, BGR→HSV, leaf foreground pre-mask |
| `phytolabs/segmentation.py` | Stage 1 GMM training + per-pixel segmentation |
| `phytolabs/features.py` | lesion area fraction, blob count, size stats |
| `phytolabs/logreg.py` | from-scratch logistic regression + SGD + scaler |
| `phytolabs/calibration.py` | suspicious band + reliability/ECE |
| `phytolabs/metrics.py` | accuracy / precision / recall / F1 / ROC-AUC (NumPy) |
| `phytolabs/pipeline.py` | end-to-end glue + batch feature tables |
| `phytolabs/viz.py` | lesion overlay, histograms, ROC, confusion matrix, reliability |
| `phytolabs/cli.py` | `make-synthetic`, `train-gmm`, `build-features`, `train-logreg`, `predict` |
| `phytolabs/synthetic.py` | synthetic leaves for an end-to-end smoke test |

## Install

```bash
cd phytolabs
python -m venv .venv && source .venv/bin/activate
pip install -e .          # or: pip install -r requirements.txt
```

## Quick start (no dataset required)

```bash
# 1. Generate synthetic leaves so the whole pipeline runs end-to-end.
python -m phytolabs.cli make-synthetic --data-dir data

# 2. Stage 1: fit the GMM and inspect the component→class mapping.
python -m phytolabs.cli train-gmm --data-dir data --k 4

# 3. Build per-image feature tables (train + val).
python -m phytolabs.cli build-features --data-dir data

# 4. Stage 2: train the from-scratch logistic regression and evaluate.
python -m phytolabs.cli train-logreg

# 5. Predict on a single image (saves a lesion overlay).
python -m phytolabs.cli predict data/val/rust/rust_000.png --save-overlay outputs/overlay.png
```

The notebooks in `notebooks/` walk through each stage with inline visuals and
fall back to synthetic data if no real dataset is present. To run the full
pipeline on Google Colab (which avoids local environment issues), use
[`notebooks/colab_phytolabs.ipynb`](notebooks/colab_phytolabs.ipynb).

## Using a real dataset

PlantVillage does not include wheat, so use one of:

- **Kaggle "Wheat Leaf Dataset"** (single leaf, close view, clean background) —
  *recommended* for color segmentation.
- **CGIAR Computer Vision for Crop Disease**
  (`shadabhussain/cgiar-computer-vision-for-crop-disease`) — multiple leaves per
  image and complex backgrounds; harder for HSV/GMM (stretch goal).

See [`scripts/download_data.sh`](scripts/download_data.sh) for Kaggle CLI steps.
After downloading, reshape into the expected layout:

```
data/
  train/  healthy/*.jpg   rust/*.jpg
  val/    healthy/*.jpg   rust/*.jpg
```

with the helper:

```bash
python -m scripts.reshape_data \
    --healthy-src data/raw/<dataset>/Healthy \
    --rust-src    data/raw/<dataset>/leaf_rust \
    --out data --val-fraction 0.2
```

Then run the same `train-gmm → build-features → train-logreg → predict` steps.

> Note on color tuning: the GMM component→class mapping in
> [`segmentation.py`](src/phytolabs/segmentation.py) uses HSV hue ranges tuned
> for orange/brown rust vs. green tissue. On a new dataset, inspect the printed
> component HSV means from `train-gmm` and adjust `classify_component` if needed.

## Honest limitations
- Image-level only: no per-region/segmentation accuracy is reported or claimed.
- Color-based segmentation is sensitive to lighting, white balance, and
  backgrounds; the clean single-leaf dataset works best.
- Single disease (brown rust). Other rusts/diseases are out of scope.
