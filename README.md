# WildWatch MLOps

End-to-end MLOps platform for wildlife camera trap image classification.
Demonstrates data versioning, experiment tracking, CI/CD, model serving,
production monitoring with drift detection, and a retraining loop.

**Stack:** Python 3.11 · PyTorch · DVC · MLflow · FastAPI · Docker · GitHub Actions · Streamlit
**Deployment target:** GCP Cloud Run

---

## Project status

This project is being built in six phases. Each phase ends with a review gate.

| Phase | Scope                                      | Status         |
| ----- | ------------------------------------------ | -------------- |
| 1     | Scaffolding & data pipeline (DVC)          | **Complete**   |
| 2     | Experiment tracking & model training       | Not started    |
| 3     | Model registry & CI/CD (GitHub Actions)    | Not started    |
| 4     | Model serving (FastAPI + Docker)           | Not started    |
| 5     | Monitoring & drift detection (Streamlit)   | Not started    |
| 6     | Simulated drift + retraining demo          | Not started    |

See `wildwatch_mlops_prompt.md` for the full specification.

---

## Phase 1 — Data pipeline (current)

A three-stage DVC pipeline produces a validated, split dataset from raw
images. The current dataset is a **synthetic placeholder** (8 classes ×
80 images = 640 images) so the downstream pipeline can be exercised
without requiring an external dataset download. A `TODO` block in
`src/data/download.py` lists real-dataset swap-in candidates
(Caltech Camera Traps, Animals-10, iNaturalist).

### Pipeline stages (`dvc.yaml`)

```
download   src/data/download.py    →  data/raw/<class>/*.jpg
preprocess src/data/preprocess.py  →  data/processed/{train,val,test}/<class>/*.jpg
validate   src/data/validate.py    →  reports/data_validation.json
```

- **download** — generates the synthetic dataset (deterministic per-class
  color anchors + gradient + noise, so there's learnable signal for Phase 2).
- **preprocess** — resizes to 224×224 and produces stratified 70/15/15
  splits per class, with per-class deterministic shuffling.
- **validate** — checks image readability, min samples per class, and
  class imbalance ratio. Writes a structured JSON report and exits
  non-zero on failure.

All thresholds (split ratios, min samples, imbalance cap, seed, image
size, class list) live in `params.yaml`. No magic numbers in source.

### Repo layout (Phase 1)

```
wildwatch/
├── params.yaml                 central config for all phases
├── pyproject.toml              Python 3.11.*, deps + ruff/mypy/pytest
├── dvc.yaml                    download → preprocess → validate
├── dvc.lock                    pinned stage hashes
├── src/
│   ├── data/
│   │   ├── download.py         synthetic generator (TODO: real dataset)
│   │   ├── preprocess.py       resize + stratified splits
│   │   └── validate.py         readability / min-samples / imbalance checks
│   ├── training/               (Phase 2)
│   ├── serving/                (Phase 4)
│   └── monitoring/             (Phase 5)
├── tests/                      (tests arrive alongside their phase)
├── scripts/                    (Phase 3, 6)
├── notebooks/                  (Phase 2)
├── data/                       DVC-tracked, not git-tracked
│   ├── raw/                    (produced by download stage)
│   └── processed/              (produced by preprocess stage)
├── models/                     (Phase 2)
└── reports/
    └── data_validation.json    latest validation report
```

### Setup

Prerequisites: Python 3.11 (verified against 3.11.9 via the `py` launcher
on Windows), git, and — optionally — a DVC remote. Local-only DVC works
out of the box for now.

```bash
# Create an isolated Python 3.11 environment
py -3.11 -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# Phase 1 minimal deps (avoids pulling torch until Phase 2)
pip install numpy pillow pyyaml dvc

# Phase 2+ will install the full project:
# pip install -e ".[dev]"
```

### Running the Phase 1 pipeline

```bash
dvc init            # first time only
dvc repro           # run all three stages
dvc repro validate  # run just the validate stage (re-uses cached earlier stages)
```

Expected output on a clean run against the synthetic dataset:

```
download   → 640 images across 8 classes in data/raw/
preprocess → train=448 val=96 test=96 (stratified, 56/12/12 per class)
validate   → PASSED: classes=8 total=640 imbalance=1.00 corrupt=0
```

Inspect `reports/data_validation.json` for the full structured report.

### Configuration

All Phase 1 behavior is controlled by `params.yaml`:

- `seed` — master RNG seed (deterministic synthetic data + splits).
- `dataset.classes`, `dataset.samples_per_class` — synthetic dataset shape.
- `preprocess.image_size`, `preprocess.split_ratios` — preprocessing.
- `validate.min_samples_per_class`, `validate.max_class_imbalance_ratio`,
  `validate.fail_on_corrupt` — validation thresholds.

Because `dvc.yaml` declares each stage's `params:` dependencies, editing
a section only invalidates the stages that actually read it.

### Swapping in a real dataset

Replace the synthetic generator with a real downloader in
`src/data/download.py`. Keep the on-disk contract:

```
data/raw/<class_name>/<image_id>.jpg
```

Everything downstream keys off that convention — no other changes needed
to `preprocess.py`, `validate.py`, or `dvc.yaml` as long as the contract
holds and `dataset.classes` in `params.yaml` is updated to match.

---

## License

See `LICENSE`.
