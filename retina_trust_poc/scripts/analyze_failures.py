#!/usr/bin/env python3
"""
RetinaTrust v0.6 Failure Analysis CLI

Analyzes DeepDRiD predictions CSV to understand failure modes, calibration,
and disagreements between eye pairs. This is a research module for understanding
domain shift and model limitations, not a diagnostic tool.

Usage:
    python scripts/analyze_failures.py --input outputs/deepdrid_predictions.csv --output outputs/failure_analysis
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from retina_poc.external_validation import (  # noqa: E402
    UNGRADABLE_DR_GRADE,
    binary_probability_metrics,
    cluster_bootstrap_intervals,
)
from retina_poc.modeling import expected_calibration_error  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "outputs/deepdrid_predictions.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/failure_analysis"
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_BOOTSTRAP_SEED = 20260823
DEFAULT_CLASSIFICATION_THRESHOLD = 0.5

# Actual DeepDRiD partition names from the predictions CSV
DEEPDRID_PARTITIONS = {
    "regular-fundus-training": "training",
    "regular-fundus-validation": "validation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze DeepDRiD prediction failures for RetinaTrust v0.6 research."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to deepdrid_predictions.csv from external evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for analysis outputs (JSON, CSV, reports).",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Number of bootstrap samples for confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help="Random seed for reproducible bootstrap.",
    )
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=DEFAULT_CLASSIFICATION_THRESHOLD,
        help="Binary classification threshold for error analysis.",
    )
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    """Calculate SHA-256 hash of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_predictions_csv(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """
    Validate predictions CSV for duplicates, invalid labels, and schema.
    Raises ValueError with actionable error messages for critical issues.
    """
    required_columns = {
        "partition",
        "patient_id",
        "image_id",
        "eye",
        "dr_grade",
        "referable_dr",
        "quality_good",
        "probability_referable_dr",
        "confidence",
        "quality_score",
        "quality_gate",
    }
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    validation_result = {
        "schema_valid": True,
        "duplicate_image_ids": [],
        "invalid_dr_grades": [],
        "invalid_quality_gates": [],
        "missing_referable_dr": 0,
        "nonfinite_probabilities": [],
        "nonfinite_confidence": [],
        "invalid_threshold": False,
        "inconsistent_dr_labels": [],
        "total_rows": len(frame),
    }

    # Check for duplicate image_ids
    image_id_counts = frame["image_id"].value_counts()
    duplicates = image_id_counts[image_id_counts > 1].index.tolist()
    validation_result["duplicate_image_ids"] = duplicates
    if duplicates:
        raise ValueError(
            f"Found {len(duplicates)} duplicate image_ids: {duplicates[:5]}{'...' if len(duplicates) > 5 else ''}. "
            "Each image must have a unique identifier."
        )

    # Check for invalid DR grades (allow 0-5, where 5 is UNGRADABLE_DR_GRADE)
    valid_grades = set(range(6))
    invalid_grades = frame[~frame["dr_grade"].isin(valid_grades)]["image_id"].tolist()
    validation_result["invalid_dr_grades"] = invalid_grades
    if invalid_grades:
        raise ValueError(
            f"Found {len(invalid_grades)} records with invalid DR grades (must be 0-5): "
            f"{invalid_grades[:5]}{'...' if len(invalid_grades) > 5 else ''}"
        )

    # Check for invalid quality gates
    valid_gates = {"pass", "review", "fail"}
    invalid_gates = frame[~frame["quality_gate"].isin(valid_gates)]["image_id"].tolist()
    validation_result["invalid_quality_gates"] = invalid_gates
    if invalid_gates:
        raise ValueError(
            f"Found {len(invalid_gates)} records with invalid quality gates (must be pass/review/fail): "
            f"{invalid_gates[:5]}{'...' if len(invalid_gates) > 5 else ''}"
        )

    # Check for missing referable_dr when dr_grade is not UNGRADABLE
    eligible = frame[frame["dr_grade"] != UNGRADABLE_DR_GRADE]
    missing_count = int(eligible["referable_dr"].isna().sum())
    validation_result["missing_referable_dr"] = missing_count
    if missing_count > 0:
        raise ValueError(
            f"Found {missing_count} DR-eligible records with missing referable_dr label. "
            "All non-ungradable records must have a binary diagnostic label."
        )

    # Check for nonfinite or out-of-range probabilities
    nonfinite_probs = frame[~frame["probability_referable_dr"].apply(lambda x: np.isfinite(x) and 0 <= x <= 1)]["image_id"].tolist()
    validation_result["nonfinite_probabilities"] = nonfinite_probs
    if nonfinite_probs:
        raise ValueError(
            f"Found {len(nonfinite_probs)} records with nonfinite or out-of-range probabilities (must be 0-1): "
            f"{nonfinite_probs[:5]}{'...' if len(nonfinite_probs) > 5 else ''}"
        )

    # Check for nonfinite or out-of-range confidence
    nonfinite_conf = frame[~frame["confidence"].apply(lambda x: np.isfinite(x) and 0 <= x <= 1)]["image_id"].tolist()
    validation_result["nonfinite_confidence"] = nonfinite_conf
    if nonfinite_conf:
        raise ValueError(
            f"Found {len(nonfinite_conf)} records with nonfinite or out-of-range confidence (must be 0-1): "
            f"{nonfinite_conf[:5]}{'...' if len(nonfinite_conf) > 5 else ''}"
        )

    # Check classification threshold
    if not (0 <= threshold <= 1):
        validation_result["invalid_threshold"] = True
        raise ValueError(f"Classification threshold must be between 0 and 1, got {threshold}")

    # Check for inconsistent DR grades and binary labels
    eligible["expected_referable"] = (eligible["dr_grade"] >= 2).astype(int)
    inconsistent = eligible[eligible["referable_dr"] != eligible["expected_referable"]]["image_id"].tolist()
    validation_result["inconsistent_dr_labels"] = inconsistent
    if inconsistent:
        raise ValueError(
            f"Found {len(inconsistent)} records with inconsistent DR grades and binary labels "
            f"(dr_grade >= 2 should match referable_dr = 1): "
            f"{inconsistent[:5]}{'...' if len(inconsistent) > 5 else ''}"
        )

    return validation_result


def analyze_errors_by_dimension(
    frame: pd.DataFrame,
    dimension: str,
    threshold: float,
) -> list[dict[str, Any]]:
    """Analyze error rates stratified by a dimension (DR grade, quality, partition, gate, confidence)."""
    dr_eligible = frame[frame["dr_grade"] != UNGRADABLE_DR_GRADE].copy()
    if dr_eligible.empty:
        return []

    y = dr_eligible["referable_dr"].astype(int).to_numpy()
    probability = dr_eligible["probability_referable_dr"].astype(float).to_numpy()
    predicted = (probability >= threshold).astype(int)

    results: list[dict[str, Any]] = []

    if dimension == "dr_grade":
        for grade in range(5):
            subset = dr_eligible[dr_eligible["dr_grade"] == grade]
            if subset.empty:
                results.append({
                    "dr_grade": grade,
                    "n": 0,
                    "error_rate": None,
                    "mean_probability": None,
                })
                continue
            y_sub = subset["referable_dr"].astype(int).to_numpy()
            prob_sub = subset["probability_referable_dr"].astype(float).to_numpy()
            pred_sub = (prob_sub >= threshold).astype(int)
            expected = int(grade >= 2)
            errors = int(np.sum(pred_sub != expected))
            results.append({
                "dr_grade": grade,
                "n": int(len(subset)),
                "referable_target": expected,
                "error_rate": float(errors / len(subset)),
                "mean_probability": float(prob_sub.mean()),
            })

    elif dimension == "official_quality":
        for quality in [0, 1]:
            subset = dr_eligible[dr_eligible["quality_good"] == quality]
            if subset.empty:
                continue
            y_sub = subset["referable_dr"].astype(int).to_numpy()
            prob_sub = subset["probability_referable_dr"].astype(float).to_numpy()
            pred_sub = (prob_sub >= threshold).astype(int)
            errors = int(np.sum(pred_sub != y_sub))
            results.append({
                "official_quality_good": int(quality),
                "n": int(len(subset)),
                "error_rate": float(errors / len(subset)),
                "mean_probability": float(prob_sub.mean()),
            })

    elif dimension == "partition":
        # Use actual DeepDRiD partition names
        for partition in DEEPDRID_PARTITIONS.keys():
            subset = dr_eligible[dr_eligible["partition"] == partition]
            if subset.empty:
                continue
            y_sub = subset["referable_dr"].astype(int).to_numpy()
            prob_sub = subset["probability_referable_dr"].astype(float).to_numpy()
            pred_sub = (prob_sub >= threshold).astype(int)
            errors = int(np.sum(pred_sub != y_sub))
            results.append({
                "partition": partition,
                "n": int(len(subset)),
                "error_rate": float(errors / len(subset)),
                "mean_probability": float(prob_sub.mean()),
            })

    elif dimension == "quality_gate":
        for gate in ["pass", "review", "fail"]:
            subset = dr_eligible[dr_eligible["quality_gate"] == gate]
            if subset.empty:
                continue
            y_sub = subset["referable_dr"].astype(int).to_numpy()
            prob_sub = subset["probability_referable_dr"].astype(float).to_numpy()
            pred_sub = (prob_sub >= threshold).astype(int)
            errors = int(np.sum(pred_sub != y_sub))
            results.append({
                "quality_gate": gate,
                "n": int(len(subset)),
                "error_rate": float(errors / len(subset)),
                "mean_probability": float(prob_sub.mean()),
            })

    elif dimension == "confidence":
        # Stratify by confidence thresholds
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
        for conf_thresh in thresholds:
            subset = dr_eligible[dr_eligible["confidence"] >= conf_thresh]
            if subset.empty:
                continue
            y_sub = subset["referable_dr"].astype(int).to_numpy()
            prob_sub = subset["probability_referable_dr"].astype(float).to_numpy()
            pred_sub = (prob_sub >= threshold).astype(int)
            errors = int(np.sum(pred_sub != y_sub))
            results.append({
                "confidence_threshold": conf_thresh,
                "n": int(len(subset)),
                "error_rate": float(errors / len(subset)),
                "mean_probability": float(prob_sub.mean()),
            })

    return results


def analyze_calibration(frame: pd.DataFrame, bins: int = 10) -> dict[str, Any]:
    """
    Analyze calibration using ECE, Brier score, and reliability bins.
    For positive-class probability reliability, uses mean_predicted_positive_probability
    and observed_positive_rate (not classification accuracy).
    """
    dr_eligible = frame[frame["dr_grade"] != UNGRADABLE_DR_GRADE].copy()
    if dr_eligible.empty:
        return {"n": 0, "ece": None, "brier": None, "bins": []}

    y = dr_eligible["referable_dr"].astype(int).to_numpy()
    probability = dr_eligible["probability_referable_dr"].astype(float).to_numpy()

    # Calculate overall metrics
    ece = expected_calibration_error(y, probability, bins=bins)
    brier = float(np.mean((probability - y) ** 2))

    # Calculate reliability bins for positive-class probability
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_results = []
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        if high < 1.0:
            include = (probability >= low) & (probability < high)
        else:
            include = (probability >= low) & (probability <= high)

        if not include.any():
            bin_results.append({
                "bin": i,
                "range": [float(low), float(high)],
                "n": 0,
                "mean_predicted_positive_probability": None,
                "observed_positive_rate": None,
                "ece_contribution": 0.0,
            })
            continue

        bin_y = y[include]
        bin_prob = probability[include]
        observed_positive_rate = float(bin_y.mean())
        mean_predicted_positive_probability = float(bin_prob.mean())
        ece_contrib = float(include.mean() * abs(observed_positive_rate - mean_predicted_positive_probability))

        bin_results.append({
            "bin": i,
            "range": [float(low), float(high)],
            "n": int(include.sum()),
            "mean_predicted_positive_probability": mean_predicted_positive_probability,
            "observed_positive_rate": observed_positive_rate,
            "ece_contribution": ece_contrib,
        })

    return {
        "n": int(len(dr_eligible)),
        "ece": float(ece),
        "brier": brier,
        "bins": bin_results,
    }


def analyze_eye_pair_disagreements(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """Analyze disagreements between images of the same eye, grouped by partition/patient/eye."""
    # Group by partition, patient_id, and eye
    dr_eligible = frame[frame["dr_grade"] != UNGRADABLE_DR_GRADE].copy()
    if dr_eligible.empty:
        return {"total_pairs": 0, "disagreements": []}

    pairs: list[dict[str, Any]] = []
    for (partition, patient_id, eye), group in dr_eligible.groupby(
        ["partition", "patient_id", "eye"], sort=True
    ):
        if len(group) < 2:
            continue

        predictions = (group["probability_referable_dr"].to_numpy(float) >= threshold).astype(int)
        unique_predictions = np.unique(predictions)

        pairs.append({
            "partition": partition,
            "patient_id": str(patient_id),
            "eye": str(eye),
            "image_count": int(len(group)),
            "predictions": predictions.tolist(),
            "has_disagreement": int(len(unique_predictions) > 1),
        })

    total_pairs = len(pairs)
    disagreements = [p for p in pairs if p["has_disagreement"]]

    # Summarize disagreements by partition (use actual DeepDRiD partition names)
    disagreement_summary = []
    for partition in DEEPDRID_PARTITIONS.keys():
        partition_pairs = [p for p in pairs if p["partition"] == partition]
        partition_disagreements = [p for p in partition_pairs if p["has_disagreement"]]
        if partition_pairs:
            disagreement_summary.append({
                "partition": partition,
                "total_pairs": len(partition_pairs),
                "disagreement_pairs": len(partition_disagreements),
                "disagreement_rate": float(len(partition_disagreements) / len(partition_pairs)),
            })

    return {
        "total_pairs": total_pairs,
        "disagreement_count": len(disagreements),
        "by_partition": disagreement_summary,
        "disagreements": disagreements,  # Full list for detailed inspection
    }


def calculate_bootstrap_intervals(
    frame: pd.DataFrame,
    samples: int,
    seed: int,
    threshold: float,
) -> dict[str, Any]:
    """Calculate patient-clustered bootstrap 95% CIs for key metrics."""
    dr_eligible = frame[frame["dr_grade"] != UNGRADABLE_DR_GRADE].copy()
    if dr_eligible.empty:
        return {"n": 0, "intervals": None}

    y = dr_eligible["referable_dr"].astype(int).to_numpy()
    probability = dr_eligible["probability_referable_dr"].astype(float).to_numpy()
    clusters = dr_eligible["patient_id"].astype(str).to_numpy()

    cluster_count = int(pd.Series(clusters).nunique())
    if cluster_count < 2:
        return {
            "n": int(len(dr_eligible)),
            "cluster_count": cluster_count,
            "intervals": None,
            "reason": "At least two patient clusters are required.",
        }

    bootstrap_result = cluster_bootstrap_intervals(
        y,
        probability,
        clusters,
        threshold=threshold,
        samples=samples,
        seed=seed,
    )

    return {
        "n": int(len(dr_eligible)),
        "cluster_count": cluster_count,
        "intervals": bootstrap_result,
    }


def calculate_overall_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """Calculate overall point estimates for key scientific metrics."""
    dr_eligible = frame[frame["dr_grade"] != UNGRADABLE_DR_GRADE].copy()
    if dr_eligible.empty:
        return {"n": 0, "metrics": None}

    y = dr_eligible["referable_dr"].astype(int).to_numpy()
    probability = dr_eligible["probability_referable_dr"].astype(float).to_numpy()

    metrics = binary_probability_metrics(y, probability, threshold=threshold)

    return {
        "n": int(len(dr_eligible)),
        "metrics": metrics,
    }


def generate_bulgarian_report(result: dict[str, Any]) -> str:
    """Generate a scientifically readable report in Bulgarian."""
    lines = [
        "# RetinaTrust v0.6 — Анализ на грешките при външна оценка",
        "",
        f"**Дата на генериране:** {result.get('generated_at_utc', 'N/A')}",
        f"**Входен файл:** {result.get('input_file', 'N/A')}",
        f"**SHA-256 на входния файл:** {result.get('input_sha256', 'N/A')}",
        f"**Общо изображения:** {result.get('total_images', 0)}",
        f"**Общо пациенти:** {result.get('total_patients', 0)}",
        f"**DR-допустими изображения:** {result.get('dr_eligible_images', 0)}",
        "",
        "## Научна декларация",
        "",
        "- Този анализ е изследователски proof-of-concept и не е медицинско изделие.",
        "- Моделът и праговете са замразени (RetinaTrust v0.3), без обучение върху DeepDRiD.",
        "- Резултатите са post-hoc изследователски анализ, не първоначален протокол.",
        "- Не се правят клинични заключения за безопасност или ефективност.",
        "",
        "## Общи метрики (point estimates)",
        "",
    ]

    overall = result.get("overall_metrics", {}).get("metrics", {})
    if overall:
        lines.extend([
            f"- AUROC: {overall.get('auroc', 'N/A'):.4f}",
            f"- Accuracy: {overall.get('accuracy', 'N/A'):.4f}",
            f"- Sensitivity: {overall.get('sensitivity', 'N/A'):.4f}",
            f"- Specificity: {overall.get('specificity', 'N/A'):.4f}",
            f"- Balanced accuracy: {overall.get('balanced_accuracy', 'N/A'):.4f}",
            f"- Macro F1: {overall.get('macro_f1', 'N/A'):.4f}",
            f"- Brier score: {overall.get('brier', 'N/A'):.4f}",
            f"- NLL: {overall.get('nll', 'N/A'):.4f}",
            f"- ECE: {overall.get('ece', 'N/A'):.4f}",
            f"- Confusion matrix: {overall.get('confusion_matrix', 'N/A')}",
            "",
        ])

    lines.extend([
        "## Анализ на грешките по DR степен",
        "",
    ])

    for item in result.get("errors_by_dr_grade", []):
        if item["n"] == 0:
            lines.append(f"- DR степен {item['dr_grade']}: няма данни")
        else:
            lines.append(
                f"- DR степен {item['dr_grade']}: n={item['n']}, "
                f"пропорция грешки={item['error_rate']:.4f}, "
                f"средна вероятност={item['mean_probability']:.4f}"
            )

    lines.extend([
        "",
        "## Анализ на грешките по официално качество",
        "",
    ])

    for item in result.get("errors_by_quality", []):
        quality_label = "годни" if item["official_quality_good"] == 1 else "негодни"
        lines.append(
            f"- Официално {quality_label}: n={item['n']}, "
            f"пропорция грешки={item['error_rate']:.4f}, "
            f"средна вероятност={item['mean_probability']:.4f}"
        )

    lines.extend([
        "",
        "## Анализ на грешките по partition",
        "",
    ])

    for item in result.get("errors_by_partition", []):
        lines.append(
            f"- {item['partition']}: n={item['n']}, "
            f"пропорция грешки={item['error_rate']:.4f}, "
            f"средна вероятност={item['mean_probability']:.4f}"
        )

    lines.extend([
        "",
        "## Анализ на грешките по quality gate",
        "",
    ])

    for item in result.get("errors_by_gate", []):
        lines.append(
            f"- Gate {item['quality_gate']}: n={item['n']}, "
            f"пропорция грешки={item['error_rate']:.4f}, "
            f"средна вероятност={item['mean_probability']:.4f}"
        )

    lines.extend([
        "",
        "## Калибрация",
        "",
        f"- ECE ({result.get('calibration', {}).get('n', 0)} изображения): "
        f"{result.get('calibration', {}).get('ece', 'N/A'):.4f}",
        f"- Brier score: {result.get('calibration', {}).get('brier', 'N/A'):.4f}",
        "",
        "## Несъгласия между изображения на едно око",
        "",
        f"- Общо двойки: {result.get('eye_disagreements', {}).get('total_pairs', 0)}",
        f"- Двойки с несъгласие: {result.get('eye_disagreements', {}).get('disagreement_count', 0)}",
        "",
    ])

    for item in result.get("eye_disagreements", {}).get("by_partition", []):
        lines.append(
            f"- {item['partition']}: {item['disagreement_pairs']}/{item['total_pairs']} "
            f"({item['disagreement_rate']:.2%}) двойки с несъгласие"
        )

    lines.extend([
        "",
        "## 95% доверителни интервали (patient-cluster bootstrap)",
        "",
    ])

    bootstrap = result.get("bootstrap_intervals", {})
    if bootstrap.get("intervals"):
        intervals = bootstrap["intervals"]["intervals"]
        for metric, ci in intervals.items():
            if ci.get("95_ci"):
                lines.append(f"- {metric}: [{ci['95_ci'][0]:.4f}, {ci['95_ci'][1]:.4f}]")
    else:
        lines.append(f"- Няма изчислени интервали: {bootstrap.get('reason', 'N/A')}")

    lines.extend([
        "",
        "## Изходни файлове",
        "",
        f"- JSON: {result.get('output_json', 'N/A')}",
        f"- CSV с детайли: {result.get('output_csv', 'N/A')}",
    ])

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    # Validate input file exists
    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Calculate input file SHA-256
    input_sha256 = sha256_file(args.input)

    # Load predictions
    frame = pd.read_csv(args.input)
    print(f"Loaded {len(frame)} predictions from {args.input}")

    # Validate data with actionable error messages
    validation = validate_predictions_csv(frame, args.classification_threshold)
    print(f"Validation passed: {validation['total_rows']} rows")

    # Perform analyses
    print("Running error analysis by DR grade...")
    errors_by_dr_grade = analyze_errors_by_dimension(frame, "dr_grade", args.classification_threshold)

    print("Running error analysis by official quality...")
    errors_by_quality = analyze_errors_by_dimension(frame, "official_quality", args.classification_threshold)

    print("Running error analysis by partition...")
    errors_by_partition = analyze_errors_by_dimension(frame, "partition", args.classification_threshold)

    print("Running error analysis by quality gate...")
    errors_by_gate = analyze_errors_by_dimension(frame, "quality_gate", args.classification_threshold)

    print("Running error analysis by confidence...")
    errors_by_confidence = analyze_errors_by_dimension(frame, "confidence", args.classification_threshold)

    print("Running calibration analysis...")
    calibration = analyze_calibration(frame, bins=10)

    print("Running eye-pair disagreement analysis...")
    eye_disagreements = analyze_eye_pair_disagreements(frame, args.classification_threshold)

    print("Running bootstrap confidence intervals...")
    bootstrap_intervals = calculate_bootstrap_intervals(
        frame,
        args.bootstrap_samples,
        args.bootstrap_seed,
        args.classification_threshold,
    )

    print("Calculating overall metrics...")
    overall_metrics = calculate_overall_metrics(frame, args.classification_threshold)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Write CSV with detailed disagreements
    csv_path = None
    if eye_disagreements["disagreements"]:
        disagreements_df = pd.DataFrame(eye_disagreements["disagreements"])
        csv_path = args.output_dir / "eye_pair_disagreements.csv"
        disagreements_df.to_csv(csv_path, index=False)
        print(f"Wrote disagreements CSV to {csv_path}")

    # Write Bulgarian report
    report = generate_bulgarian_report({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(args.input),
        "input_sha256": input_sha256,
        "total_images": int(len(frame)),
        "total_patients": int(frame["patient_id"].nunique()),
        "dr_eligible_images": int(len(frame[frame["dr_grade"] != UNGRADABLE_DR_GRADE])),
        "overall_metrics": overall_metrics,
        "errors_by_dr_grade": errors_by_dr_grade,
        "errors_by_quality": errors_by_quality,
        "errors_by_partition": errors_by_partition,
        "errors_by_gate": errors_by_gate,
        "calibration": calibration,
        "eye_disagreements": eye_disagreements,
        "bootstrap_intervals": bootstrap_intervals,
        "output_json": str(args.output_dir / "failure_analysis.json"),
        "output_csv": str(csv_path) if csv_path else None,
    })
    report_path = args.output_dir / "FAILURE_ANALYSIS_REPORT_BG.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote Bulgarian report to {report_path}")

    # Compile final result with all metadata
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(args.input),
        "input_sha256": input_sha256,
        "total_images": int(len(frame)),
        "total_patients": int(frame["patient_id"].nunique()),
        "dr_eligible_images": int(len(frame[frame["dr_grade"] != UNGRADABLE_DR_GRADE])),
        "classification_threshold": args.classification_threshold,
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "validation": validation,
        "overall_metrics": overall_metrics,
        "errors_by_dr_grade": errors_by_dr_grade,
        "errors_by_quality": errors_by_quality,
        "errors_by_partition": errors_by_partition,
        "errors_by_gate": errors_by_gate,
        "errors_by_confidence": errors_by_confidence,
        "calibration": calibration,
        "eye_disagreements": eye_disagreements,
        "bootstrap_intervals": bootstrap_intervals,
        "output_json": str(args.output_dir / "failure_analysis.json"),
        "output_csv": str(csv_path) if csv_path else None,
        "output_report": str(report_path),
        "model_frozen": True,
        "model_version": "v0.3",
        "analysis_type": "post_hoc_research",
        "designated_for": "failure_mode_analysis_only",
    }

    # Write JSON after all paths and metadata are populated
    json_path = args.output_dir / "failure_analysis.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Wrote JSON to {json_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("FAILURE ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total images: {result['total_images']}")
    print(f"Total patients: {result['total_patients']}")
    print(f"DR-eligible images: {result['dr_eligible_images']}")

    if overall_metrics.get("metrics"):
        metrics = overall_metrics["metrics"]
        print(f"AUROC: {metrics.get('auroc', 'N/A'):.4f}")
        print(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
        print(f"Sensitivity: {metrics.get('sensitivity', 'N/A'):.4f}")
        print(f"Specificity: {metrics.get('specificity', 'N/A'):.4f}")
        print(f"Brier score: {metrics.get('brier', 'N/A'):.4f}")
        print(f"ECE: {metrics.get('ece', 'N/A'):.4f}")

    print(f"Eye pairs with disagreement: {eye_disagreements['disagreement_count']}/{eye_disagreements['total_pairs']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
