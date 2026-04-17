# WildWatch MLOps — Claude Code Implementation Prompt

## Project Overview

Build **WildWatch MLOps**, a full ML lifecycle management platform for wildlife camera trap image classification. This project demonstrates end-to-end MLOps: data versioning, experiment tracking, CI/CD for model deployment, model serving via API, production monitoring with drift detection, and a simulated retraining loop.

**Domain:** Wildlife camera trap image classification (use publicly available datasets).
**Stack:** Python, PyTorch, DVC, MLflow, FastAPI, Docker, GitHub Actions, Streamlit.
**Deployment target:** GCP Cloud Run (containerized FastAPI service).

---

## Repo Structure

```
wildwatch-mlops/
├── README.md
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── dvc.lock
├── params.yaml                    # Central config (hyperparams, paths, thresholds)
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Lint, test, type check on every PR
│       └── retrain.yml            # Scheduled/manual retraining pipeline
├── data/
│   ├── raw/                       # DVC-tracked raw images
│   ├── processed/                 # DVC-tracked cleaned splits
│   └── .gitignore
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py            # Script to fetch dataset (Caltech Camera Traps subset or similar)
│   │   ├── preprocess.py          # Resize, normalize, split
│   │   └── validate.py            # Data validation checks (schema, corruption, class balance)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── model.py               # Model definition (EfficientNet-B0 fine-tune)
│   │   ├── train.py               # Training loop with MLflow logging
│   │   ├── evaluate.py            # Evaluation metrics, confusion matrix generation
│   │   └── config.py              # Pydantic config models for training params
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── app.py                 # FastAPI application
│   │   ├── schemas.py             # Request/response Pydantic models
│   │   └── inference.py           # Model loading and prediction logic
│   └── monitoring/
│       ├── __init__.py
│       ├── logger.py              # Prediction logging (input hash, output, confidence, timestamp)
│       ├── drift.py               # Drift detection (PSI, KS test on confidence distributions)
│       ├── alerts.py              # Alert dispatcher (log warning, optional Slack webhook)
│       └── dashboard.py           # Streamlit dashboard for monitoring
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory data analysis
│   └── 02_experiment_analysis.ipynb  # MLflow experiment comparison
├── tests/
│   ├── __init__.py
│   ├── test_data_validation.py
│   ├── test_training.py
│   ├── test_serving.py
│   ├── test_monitoring.py
│   └── test_sanity_inference.py   # Smoke test: known image produces expected class
├── scripts/
│   ├── register_model.py          # Promote model in MLflow registry
│   └── simulate_drift.py          # Introduce distribution shift for demo
└── mlruns/                        # MLflow tracking (gitignored)
```

---

## Implementation Phases

Work through these phases sequentially. **Each phase ends with a GATE.** At each gate, stop execution, summarize what was built, list the files created/modified, and wait for my review and approval before proceeding.

---

### Phase 1: Project Scaffolding & Data Pipeline

**Goal:** Set up the repo structure, dependency management, data download, preprocessing, validation, and DVC tracking.

**Tasks:**

1. Initialize the repo with the directory structure above.
2. Create `pyproject.toml` with dependencies:
   - Core: `torch`, `torchvision`, `timm`, `pillow`, `numpy`, `pandas`
   - MLOps: `dvc`, `mlflow`
   - Serving: `fastapi`, `uvicorn`, `python-multipart`
   - Monitoring: `streamlit`, `scipy`, `plotly`
   - Dev: `pytest`, `ruff`, `mypy`
3. Implement `src/data/download.py`:
   - Download a manageable wildlife image classification dataset. Use one of:
     - A curated subset of Caltech Camera Traps (via a public URL or torchvision)
     - The Animals-10 dataset from Kaggle (if accessible via direct URL)
     - Or generate a synthetic placeholder dataset of 500+ images across 5-8 classes with realistic directory structure, with a clear TODO comment noting where to swap in the real dataset.
   - Save to `data/raw/` organized by class subdirectory.
4. Implement `src/data/preprocess.py`:
   - Resize images to 224x224.
   - Create stratified train/val/test splits (70/15/15).
   - Save to `data/processed/train/`, `data/processed/val/`, `data/processed/test/`.
   - Log split statistics (class counts per split).
5. Implement `src/data/validate.py`:
   - Check for corrupt/unreadable images.
   - Validate minimum samples per class (configurable threshold).
   - Check class imbalance ratio.
   - Return a structured validation report (pass/fail with details).
6. Create `dvc.yaml` pipeline:
   - Stage 1: `download` — runs `download.py`
   - Stage 2: `preprocess` — runs `preprocess.py`, depends on `data/raw/`
   - Stage 3: `validate` — runs `validate.py`, depends on `data/processed/`
7. Create `params.yaml` with all configurable values (image size, split ratios, min samples per class, etc.).
8. Add `.gitignore` entries for `data/`, `mlruns/`, `__pycache__/`, `.venv/`.

**GATE 1 — STOP HERE.**
Summarize what was built. List all files created. Confirm the DVC pipeline runs end-to-end with `dvc repro`. Wait for my review before proceeding to Phase 2.

---

### Phase 2: Experiment Tracking & Model Training

**Goal:** Train a baseline model with disciplined experiment tracking via MLflow.

**Tasks:**

1. Implement `src/training/config.py`:
   - Pydantic model for training config: learning rate, epochs, batch size, backbone (default: EfficientNet-B0 via `timm`), freeze strategy, augmentation flags.
   - Load defaults from `params.yaml`.
2. Implement `src/training/model.py`:
   - Load pretrained EfficientNet-B0 from `timm`.
   - Replace classifier head for N classes (detected from dataset).
   - Support freezing/unfreezing backbone layers.
3. Implement `src/training/train.py`:
   - Standard PyTorch training loop.
   - Data augmentation: RandomHorizontalFlip, RandomRotation, ColorJitter (togglable via config).
   - MLflow integration:
     - Log all hyperparameters at run start.
     - Log training loss and validation accuracy per epoch.
     - Log the DVC data version hash as a tag.
     - Log the final model artifact.
     - Log a confusion matrix image as an artifact.
   - Support resuming from a checkpoint.
4. Implement `src/training/evaluate.py`:
   - Compute accuracy, precision, recall, F1 (macro and per-class).
   - Generate and save confusion matrix plot.
   - Return metrics as a dictionary for MLflow logging.
5. Add a DVC stage for training in `dvc.yaml`:
   - Depends on `data/processed/` and `params.yaml`.
   - Outputs model checkpoint to `models/`.
6. Write `tests/test_training.py`:
   - Test that model forward pass works with dummy input.
   - Test that config loads correctly from params.yaml.
   - Test that evaluation metrics computation is correct on synthetic predictions.

**GATE 2 — STOP HERE.**
Summarize what was built. Confirm that a training run completes successfully with MLflow logging (even if on a tiny data subset). Show an example of logged metrics. Wait for my review before proceeding to Phase 3.

---

### Phase 3: Model Registry & CI/CD

**Goal:** Build the promotion workflow and GitHub Actions pipelines.

**Tasks:**

1. Implement `scripts/register_model.py`:
   - Accept an MLflow run ID as argument.
   - Register the model in MLflow's model registry under the name "wildwatch-classifier".
   - Transition the model through stages: None → Staging → Production.
   - Log the promotion event with timestamp and metrics snapshot.
2. Create `.github/workflows/ci.yml`:
   - Trigger on push to `main` and on PRs.
   - Steps: checkout, install dependencies, run `ruff check`, run `mypy src/`, run `pytest tests/`.
3. Create `.github/workflows/retrain.yml`:
   - Trigger: manual (`workflow_dispatch`) and weekly schedule.
   - Steps:
     - Checkout repo.
     - Pull data via DVC.
     - Run training with current params.
     - Evaluate new model against current production model metrics (read from MLflow registry).
     - If new model metrics exceed production metrics by a configurable threshold, promote to Staging.
     - Log a summary of the comparison as a workflow artifact.
   - Include a `needs: ci` dependency so tests must pass first.
4. Write `tests/test_sanity_inference.py`:
   - Load the latest production model from MLflow registry.
   - Run inference on 3-5 known test images.
   - Assert predictions are within expected classes (not exact match, but sanity check).

**GATE 3 — STOP HERE.**
Summarize what was built. Show the CI workflow YAML. Show the retraining workflow YAML. Walk through the promotion logic. Wait for my review before proceeding to Phase 4.

---

### Phase 4: Model Serving

**Goal:** Deploy the production model behind a clean FastAPI service, containerized with Docker.

**Tasks:**

1. Implement `src/serving/schemas.py`:
   - `PredictionRequest`: accepts image as `UploadFile`.
   - `PredictionResponse`: class label, confidence score, model version, timestamp, all class probabilities.
   - `HealthResponse`: status, model version, uptime.
2. Implement `src/serving/inference.py`:
   - Load model from MLflow registry (Production stage) or from a local path (fallback).
   - Preprocess input image (resize, normalize, tensor conversion).
   - Run inference, return top prediction and all class probabilities.
   - Cache the loaded model (do not reload on every request).
3. Implement `src/serving/app.py`:
   - `GET /health` — returns model version, status, uptime.
   - `POST /predict` — accepts image, returns prediction response.
   - `GET /metrics` — returns basic request count and average latency.
   - Add CORS middleware.
   - Add request logging middleware.
4. Create `Dockerfile`:
   - Python 3.11 slim base.
   - Install production dependencies only.
   - Copy model artifact (or configure to pull from MLflow/GCS at startup).
   - Expose port 8080 (Cloud Run convention).
   - CMD: `uvicorn src.serving.app:app --host 0.0.0.0 --port 8080`.
5. Create `docker-compose.yml`:
   - Service for the FastAPI app.
   - Service for MLflow tracking server (SQLite backend, local artifact store).
   - Optional: service for Streamlit dashboard (Phase 5).
6. Write `tests/test_serving.py`:
   - Test `/health` endpoint returns 200.
   - Test `/predict` with a valid image returns expected schema.
   - Test `/predict` with invalid input returns 422.

**GATE 4 — STOP HERE.**
Summarize what was built. Confirm the Docker image builds and the API responds correctly to test requests. Show example curl commands and responses. Wait for my review before proceeding to Phase 5.

---

### Phase 5: Monitoring & Drift Detection

**Goal:** Build production monitoring — prediction logging, confidence distribution tracking, drift detection, and a live dashboard.

**Tasks:**

1. Implement `src/monitoring/logger.py`:
   - Log every prediction to a local store (SQLite or append-only CSV):
     - Timestamp, input image hash (SHA256), predicted class, confidence score, model version.
   - Provide functions to query logs by time window.
2. Integrate the prediction logger into `src/serving/app.py`:
   - Every `/predict` call logs through the logger.
3. Implement `src/monitoring/drift.py`:
   - `compute_psi(reference_distribution, current_distribution)` — Population Stability Index.
   - `compute_ks_test(reference_confidences, current_confidences)` — KS statistic and p-value.
   - `check_drift(reference_window, current_window, threshold)` — returns drift detected (bool), metrics, severity level.
   - Reference window: first N predictions after model deployment. Current window: most recent M predictions. Both N and M configurable in `params.yaml`.
4. Implement `src/monitoring/alerts.py`:
   - `dispatch_alert(drift_report)`:
     - Always log to file.
     - If `SLACK_WEBHOOK_URL` env var is set, send to Slack.
     - Include: drift metric values, severity, timestamp, model version.
5. Implement `src/monitoring/dashboard.py` (Streamlit):
   - Prediction volume over time (line chart).
   - Confidence distribution histogram (current vs. reference, overlaid).
   - Per-class prediction frequency (bar chart).
   - Drift metric time series (PSI and KS stat over sliding windows).
   - Alert log table.
   - Model version indicator.
6. Write `tests/test_monitoring.py`:
   - Test PSI computation with known distributions (identical should be ~0, shifted should be high).
   - Test KS test with identical vs. different distributions.
   - Test logger writes and reads correctly.
   - Test drift check returns correct bool for threshold.

**GATE 5 — STOP HERE.**
Summarize what was built. Show the Streamlit dashboard running with synthetic prediction data. Demonstrate that drift detection correctly identifies a simulated shift. Wait for my review before proceeding to Phase 6.

---

### Phase 6: Simulated Drift & Retraining Demo

**Goal:** Tie the full loop together. Simulate distribution shift, show the monitoring catches it, and walk through the retraining response.

**Tasks:**

1. Implement `scripts/simulate_drift.py`:
   - Take the test set and create a "shifted" version:
     - Option A: Heavily oversample 1-2 classes, undersample others.
     - Option B: Apply domain-shift augmentations (heavy brightness/contrast changes, simulating night images or weather).
     - Option C: Both.
   - Run the shifted images through the `/predict` endpoint in batch.
   - Log all predictions through the normal logging pipeline.
2. After running the drift simulation:
   - Show the monitoring dashboard detecting the shift (PSI spike, KS test significance).
   - Show the alert being dispatched.
3. Document the retraining response:
   - Add the shifted data to the DVC-tracked dataset.
   - Trigger the retraining workflow (manually or via the GitHub Action).
   - Show the new model being evaluated against the old one.
   - Show promotion to Staging (if improved).
4. Write a comprehensive `README.md`:
   - Project motivation and architecture overview.
   - Architecture diagram (can be a Mermaid diagram in the README).
   - Setup instructions (prerequisites, installation, configuration).
   - How to run each phase (data pipeline, training, serving, monitoring).
   - How to reproduce the drift simulation demo.
   - Tech stack summary with links.
   - Results: example metrics, dashboard screenshots (placeholder notes for where to add them).
   - Future work section (automated closed-loop retraining, A/B testing, cloud-native monitoring with Prometheus/Grafana).

**GATE 6 — STOP HERE.**
Summarize the complete project. Confirm the full loop works: data → train → serve → monitor → detect drift → retrain. Walk through the README. This is the final review before the project is considered complete.

---

## General Guidelines

- **Code quality:** Use type hints throughout. Follow PEP 8. Use `ruff` for linting.
- **Configuration:** All magic numbers and thresholds go in `params.yaml`. No hardcoded values in source code.
- **Error handling:** All modules should have proper error handling with informative messages. No bare `except:` blocks.
- **Logging:** Use Python's `logging` module consistently. No `print()` statements in source code.
- **Docstrings:** Every public function and class gets a docstring.
- **Reproducibility:** Random seeds configurable via `params.yaml`. All experiments should be reproducible from their MLflow entry + DVC data hash.
- **Keep it real:** This is a portfolio project, not a production system. Prefer clarity over over-engineering. Comment on where you would do things differently in production (e.g., "In production, this would use a managed message queue instead of polling").
