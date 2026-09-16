# USOVA3D — 3D Ovarian Follicle Detection

Automated detection, measurement, and IVF-retrieval-maturity classification
of ovarian follicles from 3D transvaginal ultrasound volumes, built on the
USOVA3D research dataset.

**Pipeline:** raw 3D ultrasound volume → segment the ovary → detect follicle
candidates inside the ovary → measure each follicle's volume/diameter →
classify by clinical size thresholds (immature / optimal for retrieval /
post-mature), mirroring how antral follicle count (AFC) is reported
clinically.

An interactive local web app (`app.py`) lets you upload a `.vtk` volume and
view the results slice-by-slice with a measurements table.

> **Status:** research/coursework project, single contributor. Ovary
> segmentation and follicle detection are validated with cross-validation
> (see `RESULTS_LOG.md`); the web app currently wires up the classical
> Random Forest path only. A 2D U-Net alternative for follicle segmentation
> is being developed on Kaggle (`follicle_segmentation_Unet.ipynb`) for
> integration as a second option in the app.

## How it works

| Stage | Method | Code |
|---|---|---|
| Ovary segmentation | Per-voxel Random Forest on multiscale intensity/gradient/texture features (4 scales) + a positional feature | `src/features.py`, `src/ovary_cv.py` |
| Follicle detection | Gaussian smoothing → intensity threshold (within ovary ROI) → morphological opening → connected components → min-size filter | `src/follicle_detection.py` |
| Ground-truth matching | Python port of the official USOVA3D `czEvaluate.m` / `AgreedBothExperts.m` greedy matching algorithm | `src/follicle_matching.py` |
| Measurement & classification | Voxel count → physical volume (via spacing) → equivalent sphere diameter → maturity class | `src/pipeline.py` |
| Alternative segmentation (experimental) | 2D U-Net (PyTorch), trained per-slice, re-stacked to 3D | `follicle_segmentation_Unet.ipynb` (Kaggle) |

Volumes are stored as legacy VTK `STRUCTURED_POINTS` files; `src/vtk_io.py`
is a small dependency-free reader/writer for that format (no VTK/ITK
install required).

See [`RESULTS_LOG.md`](RESULTS_LOG.md) for the full experiment history
(method development, cross-validation numbers, the ovary-erosion boundary
fix, and the RF-vs-U-Net comparison), and [`REFERENCES.md`](REFERENCES.md)
for the dataset, published baseline, and clinical literature this project
relies on.

## Project layout

```
app.py                          Flask web app (upload a .vtk, view results)
src/                             Core library
  vtk_io.py                       Legacy VTK reader/writer
  features.py                     Multiscale voxel feature extraction
  ovary_cv.py                     Ovary RF training + LOO cross-validation
  follicle_detection.py           Threshold/morphology follicle detection
  follicle_matching.py            Official matching/scoring algorithm port
  follicle_nested_cv.py           Nested CV for follicle hyperparameters
  full_pipeline_cv.py             Honest end-to-end CV (predicted ROI)
  pipeline.py                     run_pipeline(): segment → detect → measure → classify
scripts/                         Numbered, sequential CLI steps (01-09)
static/, templates/              Web app frontend (vanilla JS/CSS, Jinja2)
follicle_segmentation_Unet.ipynb Kaggle notebook: 2D U-Net for follicle segmentation
RESULTS_LOG.md                   Experiment log (methods + validated results)
REFERENCES.md                    Dataset, baseline paper, and clinical citations
```

`data/` and `outputs/` are gitignored — they hold the dataset, cached
features, trained models, and generated results, and are not checked in.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements-web.txt` (Flask only) is a minimal subset if you just want to
run a pre-trained model in the web app without re-running the training/CV
scripts.

### Getting the dataset

The USOVA3D dataset is **not included in this repository** — obtain it
through the official challenge channel (see `REFERENCES.md`), then:

1. Place `Training_Set_2019.zip`, `Test_Set_2019.zip`, and
   `USOVA3D_tools.zip` in `data/`.
2. `python scripts/01_extract_zips.py`

## Running the pipeline end-to-end

Scripts are numbered and meant to be run in order from the project root:

```bash
python scripts/01_extract_zips.py            # unzip the dataset into data/
python scripts/02_cache_features.py          # cache multiscale features for the 16 training volumes
python scripts/03_run_ovary_cv.py            # 16-fold LOO-CV for ovary segmentation (resumable)
python scripts/04_run_follicle_nested_cv.py  # nested CV for follicle-detection hyperparameters (resumable)
python scripts/05_train_and_save_final_model.py  # train + save the final ovary RF model
python scripts/06_run_full_pipeline_cv.py    # honest end-to-end CV using the predicted (not GT) ovary ROI
python scripts/07_run_on_test_set.py         # run on the 19 held-out test volumes, write official-format VTK + CSVs
```

`scripts/08_quick_erosion_check.py` is an informal, ad-hoc experiment
script (not a rigorous test). `scripts/09_export_slices_for_cnn.py`
exports 2D slices + masks as `.npz`, for uploading to Kaggle and training
the U-Net notebook.

Cross-validation scripts (`03`, `04`, `06`) checkpoint progress after every
fold and are safe to interrupt (Ctrl+C) and resume.

## Running the web app

Requires a trained model at `data/final_ovary_model.joblib` (produced by
`scripts/05_train_and_save_final_model.py`):

```bash
python app.py
# open http://127.0.0.1:5000
```

Upload a legacy `.vtk` volume; the app runs the full pipeline and displays
a per-slice canvas view with color-coded follicle status alongside a
measurements table.

## Known gaps / next steps

- The official USOVA3D composite scoring metric (which adds volume-ratio
  and surface-distance terms on top of matching) is not implemented —
  current sensitivity/precision/F1 numbers aren't directly comparable to
  the published baseline (see `REFERENCES.md`).
- `vol110` and `vol3` show unusually low follicle-detection precision;
  not yet root-caused.
- The web app does not yet expose a Random-Forest-vs-U-Net toggle, despite
  this being the planned design once the follicle U-Net is trained
  (`RESULTS_LOG.md`).
- No automated test suite; correctness is currently checked only via
  cross-validation metrics against expert ground truth.
- Dependency versions in `requirements.txt` are unpinned minimums, not the
  exact versions used during development (not recorded).

## License

Code in this repository is licensed under the [MIT License](LICENSE). The
USOVA3D dataset itself is a separate, third-party research dataset with its
own usage terms — see [`REFERENCES.md`](REFERENCES.md) before
redistributing any data, model weights trained on it, or derived outputs.
