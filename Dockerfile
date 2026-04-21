# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# WildWatch serving image.
# Small, CPU-only image that starts the FastAPI app on Cloud Run's port 8080.
# The model is loaded at runtime from either:
#   - the configured MLflow tracking URI (registry Production version), or
#   - a local checkpoint at /app/models/model.pt (volume-mounted or baked in
#     by a downstream image).
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# System deps for Pillow/torchvision image decoding.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libjpeg62-turbo \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# Install CPU-only PyTorch from the dedicated index first — keeps the image
# ~900MB smaller than the default CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.2,<2.5" "torchvision>=0.17,<0.20"

# The rest of the prod deps (dev extras intentionally skipped).
RUN pip install \
    "timm>=1.0.0" \
    "pillow>=10.0" \
    "numpy>=1.26,<2.0" \
    "pandas>=2.1" \
    "pyyaml>=6.0" \
    "pydantic>=2.5" \
    "mlflow>=2.13" \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.29" \
    "python-multipart>=0.0.9" \
    "scipy>=1.12"

# Application code + central config.
COPY src/ ./src/
COPY params.yaml ./

# Models dir must exist even if empty — the local-checkpoint fallback checks
# the path at runtime; a downstream build or a compose volume can populate it.
RUN mkdir -p /app/models /app/mlruns

# Run as non-root — Cloud Run ignores USER, but it's good hygiene elsewhere.
RUN useradd --create-home --uid 1001 app && chown -R app /app
USER app

EXPOSE 8080

# Single worker keeps the model in one process; Cloud Run handles horizontal
# scaling. For a multi-worker setup, prefer gunicorn+uvicorn workers and
# move model loading into each worker's startup.
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
