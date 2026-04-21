"""Compare a freshly trained run against the current Production model.

Reads ``models/train_summary.json`` (from the ``train`` DVC stage), fetches the
primary metric on the current Production model (if any), and promotes the new
run to ``Staging`` when it beats Production by at least the configured
``promotion_threshold``.

Writes ``reports/promotion_report.json`` that the retraining workflow uploads
as an artifact.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient

from scripts.register_model import register_and_promote

logger = logging.getLogger(__name__)


def _load_params(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _production_metric(
    client: MlflowClient,
    model_name: str,
    primary_metric: str,
) -> tuple[str | None, float | None]:
    """Return (run_id, metric_value) for the current Production version, or (None, None)."""
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
    except mlflow.exceptions.MlflowException:
        return None, None
    if not versions:
        return None, None
    run = client.get_run(versions[0].run_id)
    return versions[0].run_id, run.data.metrics.get(primary_metric)


def compare_and_promote(
    summary_path: Path,
    params: dict,
) -> dict:
    """Core decision logic. Returns a report dict describing the decision."""
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    registry_cfg = params.get("registry", {})
    model_name: str = registry_cfg.get("model_name", "wildwatch-classifier")
    primary_metric: str = registry_cfg.get("primary_metric", "test_accuracy")
    threshold: float = float(registry_cfg.get("promotion_threshold", 0.01))
    tracking_uri: str = params.get("training", {}).get(
        "mlflow_tracking_uri", "file:./mlruns"
    )

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    candidate_run_id: str = summary["mlflow_run_id"]
    candidate_metric = float(summary.get(primary_metric, 0.0))

    prod_run_id, prod_metric = _production_metric(client, model_name, primary_metric)

    decision: str
    promoted_version: str | None = None
    if prod_metric is None:
        # First model ever — promote to Staging (bootstrapping case).
        decision = "promote_no_production_exists"
    elif candidate_metric - prod_metric >= threshold:
        decision = "promote_exceeds_threshold"
    else:
        decision = "reject_below_threshold"

    if decision.startswith("promote"):
        event = register_and_promote(
            run_id=candidate_run_id,
            model_name=model_name,
            tracking_uri=tracking_uri,
            promote_to_production=False,
        )
        promoted_version = str(event["version"])

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model_name": model_name,
        "primary_metric": primary_metric,
        "threshold": threshold,
        "candidate": {
            "run_id": candidate_run_id,
            "metric_value": candidate_metric,
            "dvc_data_version": summary.get("dvc_data_version"),
        },
        "production": {
            "run_id": prod_run_id,
            "metric_value": prod_metric,
        },
        "delta": (
            candidate_metric - prod_metric if prod_metric is not None else None
        ),
        "decision": decision,
        "promoted_version": promoted_version,
    }

    out_path = Path("reports/promotion_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    logger.info(
        "Decision: %s  candidate=%.4f prod=%s delta=%s",
        decision,
        candidate_metric,
        f"{prod_metric:.4f}" if prod_metric is not None else "n/a",
        f"{report['delta']:+.4f}" if report["delta"] is not None else "n/a",
    )
    return report


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Compare a new training run to Production and promote if better."
    )
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("models/train_summary.json"),
        help="Path to the train_summary.json produced by the training stage.",
    )
    args = parser.parse_args()

    params = _load_params(args.params)
    report = compare_and_promote(args.summary, params)

    print(json.dumps(report, indent=2, sort_keys=True))
    # Never fail the workflow on "reject" — rejection is a valid outcome.
    sys.exit(0)


if __name__ == "__main__":
    main()
