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
- Spring wheat **brown / leaf rust only** (one disease, *Puccinia triticina*).
- **Image-level** classification. We do **not** claim per-region accuracy — the
  datasets only provide image-level labels.
- **Clean-background, close-up single-leaf images.** The color/GMM segmentation
  relies on a leaf-vs-background separation; field photos with soil break it
  (see *Honest limitations*).

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

## Results

Trained and evaluated on the **Mendeley "Wheat nitrogen deficiency and leaf
rust" dataset** (Leaf rust sub-set; Otsu-masked, black-background close-ups).
The model's own `train` folder was re-split 80/20 into train (480 imgs:
274 healthy / 206 rust) and validation (121 imgs: 69 / 52); the dataset's
provided test split is left untouched for an unbiased final number.

Validation (121 images, threshold 0.5):

| metric | value |
| --- | --- |
| accuracy | 0.942 |
| precision | 1.000 |
| recall | 0.865 |
| F1 | 0.928 |
| ROC-AUC | ≈ 0.98 |

Confusion: 69/69 healthy correct (0 false positives), 45/52 rust correct
(7 false negatives). Band split (0.45–0.55): 75 healthy / 2 suspicious /
44 diseased. The loss curve converges smoothly (~0.12) and calibration is
reasonable.

The 7 false negatives are **mild / early-stage infections** with very small
lesion area — because the features are lesion-*quantity* based, faint cases
look healthy. Lowering the decision threshold (e.g. `proba >= 0.30`) trades a
little precision for recall and recovers several of them; the threshold sweep
and the held-out test-set evaluation live in the last section of
[`notebooks/colab_phytolabs.ipynb`](notebooks/colab_phytolabs.ipynb).

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

PlantVillage does not include wheat. We use the
**[Mendeley "Wheat nitrogen deficiency and leaf rust image dataset"](https://data.mendeley.com/datasets/th422bg4yd/1)**
(Leaf rust sub-set), whose leaves are Otsu-masked onto a black background and
shot close-up — exactly the clean-background condition the GMM needs. Its
structure is `WheatLeafRust/{train,test,val}/{control,diseased}`; we map
`control → healthy` and `diseased → rust`.

> **Why not field datasets?** We first tried the Kaggle "Wheat Leaf Dataset"
> (field photos with soil backgrounds). The GMM flagged brown soil as rust,
> making the segmentation useless — confirming that this color-based method
> needs clean backgrounds.

After downloading, reshape into the expected layout:

```
data/
  train/  healthy/*.jpg   rust/*.jpg
  val/    healthy/*.jpg   rust/*.jpg
```

with the helper:

```bash
python -m scripts.reshape_data \
    --healthy-src data/raw/WheatLeafRust/train/control \
    --rust-src    data/raw/WheatLeafRust/train/diseased \
    --out data --val-fraction 0.2
```

Then run the same `train-gmm → build-features → train-logreg → predict` steps.
For a clean final number, evaluate on the untouched
`data/raw/WheatLeafRust/test` split (see the threshold-sweep + test-set cell in
the Colab notebook).

> Note on color tuning: the GMM component→class mapping in
> [`segmentation.py`](src/phytolabs/segmentation.py) uses HSV hue ranges tuned
> for orange/brown rust vs. green tissue. On a new dataset, inspect the printed
> component HSV means from `train-gmm` and adjust `classify_component` if needed.

## Honest limitations
- **Image-level only**: no per-region/segmentation accuracy is reported or
  claimed.
- **Mild / early infections are under-detected.** The features are
  lesion-*quantity* based, so faint cases with tiny lesion area look healthy —
  this is the source of the validation false negatives. A lower decision
  threshold recovers some at a small precision cost.
- **Backgrounds must be clean.** Color-based GMM segmentation is sensitive to
  lighting, white balance, and especially backgrounds: on field photos with
  soil, brown earth is mistaken for rust and the segmentation fails. The
  Otsu-masked single-leaf dataset is what makes this method work.
- **Single disease** (brown / leaf rust). Other rusts/diseases are out of scope.
