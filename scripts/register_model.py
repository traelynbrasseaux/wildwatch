"""Promote an MLflow run's model artifact through the registry.

Flow:

1. Register the ``runs:/<run_id>/model`` artifact under ``--name`` (default
   from ``params.yaml:registry.model_name``).
2. Transition the new version to ``Staging``.
3. Optionally (``--promote-to-production``) move it to ``Production`` and
   archive older Production versions.
4. Append an audit line to ``reports/registry_events.jsonl`` capturing who
   did what when, plus a snapshot of the run's primary metrics.

MLflow 3.x has soft-deprecated stage transitions in favour of aliases +
tags, but the stage API still works and maps cleanly onto the common
"dev → staging → prod" portfolio narrative. For a production system we would
switch to :func:`MlflowClient.set_registered_model_alias`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

EVENT_LOG_PATH = Path("reports/registry_events.jsonl")


def _load_params(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _fetch_run_metrics(client: MlflowClient, run_id: str) -> dict[str, float]:
    """Return all metrics logged on ``run_id`` as a flat dict."""
    run = client.get_run(run_id)
    return dict(run.data.metrics)


def _append_event(event: dict) -> None:
    EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVENT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def register_and_promote(
    run_id: str,
    model_name: str,
    tracking_uri: str,
    artifact_path: str = "model",
    promote_to_production: bool = False,
) -> dict:
    """Register the run's model and transition stages. Returns a summary dict."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # Ensure the registered model exists; create on first run.
    # MlflowException is raised by the file-store backend; RestException
    # by the server backend — catch the common parent.
    try:
        client.get_registered_model(model_name)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(model_name)
        logger.info("Created registered model '%s'", model_name)

    model_uri = f"runs:/{run_id}/{artifact_path}"
    logger.info("Registering %s", model_uri)
    # Suppress the "stages are deprecated" FutureWarning chain here — we
    # acknowledge it in the module docstring.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        mv = mlflow.register_model(model_uri=model_uri, name=model_name)
        version = mv.version

        # None -> Staging
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
            archive_existing_versions=False,
        )
        current_stage = "Staging"

        if promote_to_production:
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True,
            )
            current_stage = "Production"

    metrics = _fetch_run_metrics(client, run_id)

    event = {
        "timestamp": datetime.now(UTC).isoformat(),
        "action": "promote_to_production" if promote_to_production else "promote_to_staging",
        "model_name": model_name,
        "version": version,
        "run_id": run_id,
        "stage": current_stage,
        "metrics": metrics,
    }
    _append_event(event)

    logger.info(
        "Registered %s v%s → %s (run_id=%s)",
        model_name,
        version,
        current_stage,
        run_id,
    )
    return event


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Register and promote an MLflow run in the model registry."
    )
    parser.add_argument("run_id", help="MLflow run ID containing the model artifact.")
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument(
        "--name",
        default=None,
        help="Registered model name (defaults to params.yaml:registry.model_name).",
    )
    parser.add_argument(
        "--artifact-path",
        default="model",
        help="Artifact subpath where the model was logged (default: 'model').",
    )
    parser.add_argument(
        "--promote-to-production",
        action="store_true",
        help="Also move the new version to Production and archive older Production versions.",
    )
    args = parser.parse_args()

    params = _load_params(args.params)
    registry_cfg = params.get("registry", {})
    model_name = args.name or registry_cfg.get("model_name", "wildwatch-classifier")
    tracking_uri = params.get("training", {}).get("mlflow_tracking_uri", "file:./mlruns")

    try:
        event = register_and_promote(
            run_id=args.run_id,
            model_name=model_name,
            tracking_uri=tracking_uri,
            artifact_path=args.artifact_path,
            promote_to_production=args.promote_to_production,
        )
    except Exception as exc:
        logger.error("Registration failed: %s", exc)
        sys.exit(1)

    print(json.dumps(event, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
