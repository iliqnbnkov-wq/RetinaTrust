#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from retina_poc.external_validation import (  # noqa: E402
    UNGRADABLE_DR_GRADE,
    aggregate_eye_predictions,
    aggregate_patient_predictions,
    binary_probability_metrics,
    cluster_bootstrap_intervals,
    grade_stratified_errors,
    load_deepdrid_records,
    predict_external_record,
    quality_gate_metrics,
    sha256_file,
    verify_protocol_lock,
)
from retina_poc.modeling import load_bundle  # noqa: E402


DEFAULT_MODEL = PROJECT_ROOT / "artifacts/v0.3/retina_baseline.joblib"
DEFAULT_PROTOCOL_LOCK = PROJECT_ROOT / "artifacts/v0.4/external_protocol_lock.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/deepdrid_external"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen zero-shot DeepDRiD external-validation protocol."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="DeepDRiD repository root or its regular_fundus_images directory.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--protocol-lock", type=Path, default=DEFAULT_PROTOCOL_LOCK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _evaluate_subset(
    frame: pd.DataFrame,
    *,
    denominator: int,
    threshold: float,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    eligible = frame[frame["referable_dr"].notna()].copy()
    if eligible.empty:
        return {
            "n": 0,
            "coverage_of_primary_denominator": 0.0,
            "metrics": None,
            "patient_clustered_bootstrap": None,
        }
    y = eligible["referable_dr"].astype(int).to_numpy()
    probability = eligible["probability_referable_dr"].astype(float).to_numpy()
    clusters = eligible["patient_id"].astype(str).to_numpy()
    cluster_count = int(pd.Series(clusters).nunique())
    bootstrap_result = (
        cluster_bootstrap_intervals(
            y,
            probability,
            clusters,
            threshold=threshold,
            samples=samples,
            seed=seed,
        )
        if cluster_count >= 2
        else {
            "method": "patient-clustered nonparametric bootstrap",
            "cluster_count": cluster_count,
            "samples_requested": samples,
            "seed": seed,
            "intervals": None,
            "reason": "At least two patient clusters are required.",
        }
    )
    return {
        "n": int(len(eligible)),
        "coverage_of_primary_denominator": float(len(eligible) / denominator),
        "metrics": binary_probability_metrics(y, probability, threshold),
        "patient_clustered_bootstrap": bootstrap_result,
    }


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.model)
    protocol = verify_protocol_lock(args.protocol_lock, PROJECT_ROOT, bundle, args.model)
    bootstrap = protocol["bootstrap"]
    samples = int(bootstrap["samples"])
    seed = int(bootstrap["seed"])
    threshold = float(protocol["thresholds"]["classification"])

    records, label_paths = load_deepdrid_records(args.dataset_root)
    predictions: list[dict[str, Any]] = []
    total = len(records)
    for index, record in enumerate(records, start=1):
        predictions.append(predict_external_record(record, bundle))
        if index % 25 == 0 or index == total:
            print(f"[DeepDRiD] images {index}/{total}", flush=True)
    frame = pd.DataFrame(predictions)

    dr_eligible = frame[frame["dr_grade"] != UNGRADABLE_DR_GRADE].copy()
    denominator = int(len(dr_eligible))
    all_images = _evaluate_subset(
        dr_eligible,
        denominator=denominator,
        threshold=threshold,
        samples=samples,
        seed=seed,
    )
    pass_only = _evaluate_subset(
        dr_eligible[dr_eligible["quality_gate"] == "pass"],
        denominator=denominator,
        threshold=threshold,
        samples=samples,
        seed=seed,
    )
    autonomous_only = _evaluate_subset(
        dr_eligible[dr_eligible["autonomous_eligible"]],
        denominator=denominator,
        threshold=threshold,
        samples=samples,
        seed=seed,
    )

    eye_frame = aggregate_eye_predictions(frame)
    eye_eligible = eye_frame[eye_frame["referable_dr"].notna()].copy()
    eye_result = _evaluate_subset(
        eye_eligible,
        denominator=int(len(eye_eligible)),
        threshold=threshold,
        samples=samples,
        seed=seed,
    )
    patient_frame = aggregate_patient_predictions(frame)
    patient_eligible = patient_frame[patient_frame["referable_dr"].notna()].copy()
    patient_result = _evaluate_subset(
        patient_eligible,
        denominator=int(len(patient_eligible)),
        threshold=threshold,
        samples=samples,
        seed=seed,
    )

    label_manifest = [
        {
            "partition": path.parent.name,
            "filename": path.name,
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in label_paths
    ]
    result = {
        "schema_version": 1,
        "status": "executed_zero_shot_external_evaluation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "DeepDRiD regular fundus",
            "partitions": ["training", "validation"],
            "images": int(len(frame)),
            "patients": int(frame["patient_id"].nunique()),
            "label_files": label_manifest,
            "adaptation_or_tuning_on_external_labels": False,
        },
        "protocol": {
            "lock_file": str(args.protocol_lock.relative_to(PROJECT_ROOT)),
            "lock_sha256": sha256_file(args.protocol_lock),
            "primary_unit": "single image",
            "uncertainty_unit": "patient cluster",
            "classification_threshold": threshold,
            "confidence_floor": float(protocol["thresholds"]["confidence_floor"]),
            "quality_intervention": "review or fail",
        },
        "primary_image_level_dr": all_images,
        "selective_image_level_dr": {
            "quality_pass_only": pass_only,
            "quality_pass_and_confidence_floor": autonomous_only,
        },
        "quality_gate": quality_gate_metrics(
            frame["quality_good"].astype(int), frame["quality_gate"].astype(str)
        ),
        "grade_stratified_binary_errors": grade_stratified_errors(frame, threshold),
        "secondary_aggregations": {
            "eye_level_max_of_views": eye_result,
            "patient_level_max_of_images": patient_result,
            "warning": (
                "Max aggregation is pre-specified and secondary; it is not a five-class model "
                "and may change false-positive behaviour."
            ),
        },
        "interpretation_limits": [
            "Cross-dataset performance combines population, camera, acquisition and labelling shift.",
            "It does not isolate demographic bias or establish clinical safety.",
            "Selective results must always be read with their reported coverage.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "deepdrid_predictions.csv", index=False)
    eye_frame.to_csv(args.output_dir / "deepdrid_eye_aggregation.csv", index=False)
    patient_frame.to_csv(args.output_dir / "deepdrid_patient_aggregation.csv", index=False)
    (args.output_dir / "deepdrid_external_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
