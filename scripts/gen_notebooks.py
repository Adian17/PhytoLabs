"""Generator for the demo notebooks. Run: `python scripts/gen_notebooks.py`.

Kept in the repo so the notebooks can be regenerated deterministically.
"""

from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    out = NB_DIR / name
    nbf.write(nb, out)
    print("wrote", out)


# ---------------------------------------------------------------------------
# 01 — Segmentation (Stage 1)
# ---------------------------------------------------------------------------
write(
    "01_segmentation_gmm.ipynb",
    [
        md(
            "# Stage 1 — GMM + EM (unsupervised lesion segmentation)\n\n"
            "Fit a Gaussian Mixture Model in **HSV** color space to cluster leaf "
            "pixels into **rust / green / background**, then visualize the lesion "
            "overlay (the core PhytoLabs UX).\n\n"
            "If you don't have a real dataset yet, this notebook generates a small "
            "synthetic one so everything runs end-to-end."
        ),
        code(
            "from _setup import DATA_DIR, ensure_dataset\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from phytolabs import io, segmentation, viz\n\n"
            "data_dir = ensure_dataset()"
        ),
        md("## Load training images"),
        code(
            "healthy_imgs, healthy_paths = io.load_folder(data_dir / 'train' / 'healthy')\n"
            "rust_imgs, rust_paths = io.load_folder(data_dir / 'train' / 'rust')\n"
            "all_imgs = healthy_imgs + rust_imgs\n"
            "print(f'{len(healthy_imgs)} healthy, {len(rust_imgs)} rust training images')\n\n"
            "plt.imshow(viz.bgr_to_rgb(rust_imgs[0])); plt.axis('off'); plt.title('example rust leaf');"
        ),
        md(
            "## Fit the GMM\n\n"
            "We pool subsampled HSV pixels across all training images and fit one "
            "global GMM, then label each component by its mean hue/saturation."
        ),
        code(
            "leaf_gmm = segmentation.train_gmm(all_imgs, k=4, random_state=0)\n"
            "for c in range(leaf_gmm.n_components):\n"
            "    h, s, v = leaf_gmm.gmm.means_[c]\n"
            "    print(f'component {c}: HSV mean=({h:.0f},{s:.0f},{v:.0f}) -> {leaf_gmm.label_map[c]}')"
        ),
        md(
            "## Visualize segmentation + lesion overlay\n\n"
            "Original, lesion overlay, rust mask, and green mask for a diseased leaf."
        ),
        code(
            "bgr = rust_imgs[0]\n"
            "seg = leaf_gmm.segment(bgr)\n"
            "viz.plot_segmentation(bgr, seg, title='Rust leaf segmentation')\n"
            "plt.show()"
        ),
        code(
            "# Same for a healthy leaf — expect (almost) no rust pixels.\n"
            "bgr_h = healthy_imgs[0]\n"
            "viz.plot_segmentation(bgr_h, leaf_gmm.segment(bgr_h), title='Healthy leaf segmentation')\n"
            "plt.show()"
        ),
        md(
            "## Persist the model\n\n"
            "Saved as `artifacts/gmm.joblib` for the later stages / CLI."
        ),
        code(
            "from _setup import ARTIFACTS_DIR\n"
            "leaf_gmm.save(ARTIFACTS_DIR / 'gmm.joblib')\n"
            "print('saved', ARTIFACTS_DIR / 'gmm.joblib')"
        ),
        md(
            "> **Tuning on real data:** inspect the printed component HSV means. If "
            "rust tissue is mislabeled, adjust the hue ranges in "
            "`phytolabs.segmentation.classify_component`."
        ),
    ],
)

# ---------------------------------------------------------------------------
# 02 — Features
# ---------------------------------------------------------------------------
write(
    "02_features.ipynb",
    [
        md(
            "# Stage 1 -> 2 — Per-image features\n\n"
            "Turn the segmentation masks into quantitative, image-level features:\n"
            "- `lesion_area_fraction` — rust pixels / leaf pixels\n"
            "- `blob_count` — number of distinct lesions\n"
            "- `blob_size_mean/std/max` — lesion size distribution\n"
            "- `blob_density` — lesions per 10k leaf pixels"
        ),
        code(
            "from _setup import DATA_DIR, ARTIFACTS_DIR, ensure_dataset\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from phytolabs import segmentation, pipeline, viz\n"
            "from phytolabs.features import FEATURE_NAMES\n\n"
            "data_dir = ensure_dataset()\n"
            "gmm_path = ARTIFACTS_DIR / 'gmm.joblib'\n"
            "if gmm_path.exists():\n"
            "    leaf_gmm = segmentation.LeafGMM.load(gmm_path)\n"
            "else:\n"
            "    leaf_gmm = pipeline.fit_gmm_from_dir(data_dir / 'train')\n"
            "    leaf_gmm.save(gmm_path)"
        ),
        md("## Build a feature table for the training split"),
        code(
            "X, y, paths = pipeline.build_feature_table(data_dir / 'train', leaf_gmm)\n"
            "print('X shape:', X.shape, '| positives (rust):', int(y.sum()))\n"
            "import numpy as np\n"
            "for name, col in zip(FEATURE_NAMES, X.T):\n"
            "    print(f'{name:22s} healthy_mean={col[y==0].mean():.3f}  rust_mean={col[y==1].mean():.3f}')"
        ),
        md(
            "## Feature distributions by class\n\n"
            "Good features separate healthy (blue) from rust (orange)."
        ),
        code(
            "viz.plot_feature_histograms(X, y, FEATURE_NAMES)\n"
            "plt.show()"
        ),
        md("## Save feature tables (train + val) for Stage 2"),
        code(
            "for split in ('train', 'val'):\n"
            "    Xs, ys, ps = pipeline.build_feature_table(data_dir / split, leaf_gmm)\n"
            "    out = ARTIFACTS_DIR / f'features_{split}.npz'\n"
            "    np.savez(out, X=Xs, y=ys, paths=np.array(ps, dtype=object))\n"
            "    print(f'{split}: {Xs.shape} -> {out}')"
        ),
    ],
)

# ---------------------------------------------------------------------------
# 03 — Logistic regression (Stage 2)
# ---------------------------------------------------------------------------
write(
    "03_logreg_classification.ipynb",
    [
        md(
            "# Stage 2 — Logistic Regression + SGD (from scratch)\n\n"
            "Train a logistic regression in pure NumPy (sigmoid + BCE + L2, "
            "mini-batch SGD) on the Stage 1 features, then evaluate image-level "
            "performance and calibration. A **0.45-0.55** band defines a third "
            "**suspicious** class."
        ),
        code(
            "from _setup import ARTIFACTS_DIR\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from phytolabs.logreg import LogisticRegressionSGD\n"
            "from phytolabs.features import FEATURE_NAMES\n"
            "from phytolabs import viz, metrics, calibration\n\n"
            "def load_table(split):\n"
            "    d = np.load(ARTIFACTS_DIR / f'features_{split}.npz', allow_pickle=True)\n"
            "    return d['X'], d['y']\n\n"
            "X_train, y_train = load_table('train')\n"
            "X_val, y_val = load_table('val')\n"
            "print('train', X_train.shape, '| val', X_val.shape)"
        ),
        md("## Train"),
        code(
            "model = LogisticRegressionSGD(lr=0.1, epochs=300, batch_size=16, l2=1e-3, random_state=0)\n"
            "model.fit(X_train, y_train)\n"
            "for name, w in zip(FEATURE_NAMES, model.w):\n"
            "    print(f'{name:22s} {w:+.3f}')\n"
            "print(f'{\"bias\":22s} {model.b:+.3f}')"
        ),
        code(
            "viz.plot_loss(model.loss_history)\n"
            "plt.show()"
        ),
        md("## Evaluate on the validation split"),
        code(
            "proba = model.predict_proba(X_val)\n"
            "pred = model.predict(X_val)\n"
            "report = metrics.classification_report(y_val, pred, proba)\n"
            "report"
        ),
        code(
            "viz.plot_confusion_matrix(y_val, pred); plt.show()\n"
            "viz.plot_roc(y_val, proba); plt.show()"
        ),
        md(
            "## Calibration and the suspicious band\n\n"
            "Reliability diagram + how many val images land in each band class."
        ),
        code(
            "viz.plot_reliability(y_val, proba, n_bins=5); plt.show()\n"
            "print('ECE:', calibration.expected_calibration_error(y_val, proba))\n"
            "print('band summary:', calibration.band_summary(proba, 0.45, 0.55))"
        ),
        md("## Persist the trained classifier"),
        code(
            "model.save(ARTIFACTS_DIR / 'logreg.joblib')\n"
            "print('saved', ARTIFACTS_DIR / 'logreg.joblib')"
        ),
    ],
)

# ---------------------------------------------------------------------------
# 04 — End-to-end demo
# ---------------------------------------------------------------------------
write(
    "04_end_to_end_demo.ipynb",
    [
        md(
            "# End-to-end demo\n\n"
            "Load the saved GMM + logistic regression and run the full pipeline on "
            "the validation set: probability, band label, and a lesion-overlay "
            "gallery."
        ),
        code(
            "from _setup import DATA_DIR, ARTIFACTS_DIR, ensure_dataset\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from phytolabs import io, segmentation, pipeline, viz\n"
            "from phytolabs.logreg import LogisticRegressionSGD\n\n"
            "data_dir = ensure_dataset()\n"
            "leaf_gmm = segmentation.LeafGMM.load(ARTIFACTS_DIR / 'gmm.joblib')\n"
            "model = LogisticRegressionSGD.load(ARTIFACTS_DIR / 'logreg.joblib')"
        ),
        md("## Predict on a few validation images"),
        code(
            "samples = []\n"
            "for cls in ('rust', 'healthy'):\n"
            "    for p in list(io.iter_image_paths(data_dir / 'val' / cls))[:4]:\n"
            "        result, seg, bgr = pipeline.predict_image(p, leaf_gmm, model)\n"
            "        caption = f\"{cls}: P={result['probability']:.2f} ({result['label']})\"\n"
            "        samples.append((bgr, seg['rust'], caption))\n"
            "        print(caption, '|', p.name)"
        ),
        md("## Lesion-overlay gallery (the product UX)"),
        code(
            "viz.overlay_gallery(samples, ncols=4)\n"
            "plt.show()"
        ),
        md(
            "## Scope reminder\n\n"
            "This is **image-level** brown-rust classification. The overlays are a "
            "qualitative bonus from the unsupervised GMM; we do **not** claim "
            "per-region accuracy because the datasets only provide image-level labels."
        ),
    ],
)

print("done")
