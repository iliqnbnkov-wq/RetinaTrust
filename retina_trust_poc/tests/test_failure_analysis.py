from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_failures import (  # noqa: E402
    analyze_calibration,
    analyze_eye_pair_disagreements,
    analyze_errors_by_dimension,
    calculate_bootstrap_intervals,
    generate_bulgarian_report,
    validate_predictions_csv,
)


class FailureAnalysisValidationTests(unittest.TestCase):
    def test_validate_predictions_csv_detects_duplicates(self) -> None:
        frame = pd.DataFrame({
            "partition": ["training", "training"],
            "patient_id": ["1", "1"],
            "image_id": ["1_l1", "1_l1"],  # Duplicate
            "eye": ["l", "l"],
            "dr_grade": [0, 0],
            "referable_dr": [0, 0],
            "quality_good": [1, 1],
            "probability_referable_dr": [0.1, 0.2],
            "confidence": [0.9, 0.8],
            "quality_score": [80.0, 80.0],
            "quality_gate": ["pass", "pass"],
        })
        result = validate_predictions_csv(frame)
        self.assertEqual(len(result["duplicate_image_ids"]), 1)
        self.assertIn("1_l1", result["duplicate_image_ids"])

    def test_validate_predictions_csv_detects_invalid_dr_grades(self) -> None:
        frame = pd.DataFrame({
            "partition": ["training"],
            "patient_id": ["1"],
            "image_id": ["1_l1"],
            "eye": ["l"],
            "dr_grade": [10],  # Invalid
            "referable_dr": [0],
            "quality_good": [1],
            "probability_referable_dr": [0.1],
            "confidence": [0.9],
            "quality_score": [80.0],
            "quality_gate": ["pass"],
        })
        result = validate_predictions_csv(frame)
        self.assertEqual(len(result["invalid_dr_grades"]), 1)
        self.assertIn("1_l1", result["invalid_dr_grades"])

    def test_validate_predictions_csv_detects_invalid_quality_gates(self) -> None:
        frame = pd.DataFrame({
            "partition": ["training"],
            "patient_id": ["1"],
            "image_id": ["1_l1"],
            "eye": ["l"],
            "dr_grade": [0],
            "referable_dr": [0],
            "quality_good": [1],
            "probability_referable_dr": [0.1],
            "confidence": [0.9],
            "quality_score": [80.0],
            "quality_gate": ["invalid"],  # Invalid
        })
        result = validate_predictions_csv(frame)
        self.assertEqual(len(result["invalid_quality_gates"]), 1)
        self.assertIn("1_l1", result["invalid_quality_gates"])

    def test_validate_predictions_csv_detects_missing_referable_dr(self) -> None:
        frame = pd.DataFrame({
            "partition": ["training"],
            "patient_id": ["1"],
            "image_id": ["1_l1"],
            "eye": ["l"],
            "dr_grade": [0],  # Not UNGRADABLE, should have referable_dr
            "referable_dr": [None],  # Missing
            "quality_good": [1],
            "probability_referable_dr": [0.1],
            "confidence": [0.9],
            "quality_score": [80.0],
            "quality_gate": ["pass"],
        })
        result = validate_predictions_csv(frame)
        self.assertEqual(result["missing_referable_dr"], 1)

    def test_validate_predictions_csv_requires_schema(self) -> None:
        frame = pd.DataFrame({
            "partition": ["training"],
            "patient_id": ["1"],
            # Missing required columns
        })
        with self.assertRaises(ValueError):
            validate_predictions_csv(frame)


class FailureAnalysisErrorTests(unittest.TestCase):
    def test_errors_by_dr_grade_returns_all_grades(self) -> None:
        frame = pd.DataFrame({
            "partition": ["training"] * 5,
            "patient_id": ["1", "2", "3", "4", "5"],
            "image_id": ["1_l1", "2_l1", "3_l1", "4_l1", "5_l1"],
            "eye": ["l"] * 5,
            "dr_grade": [0, 1, 2, 3, 4],
            "referable_dr": [0, 0, 1, 1, 1],
            "quality_good": [1] * 5,
            "probability_referable_dr": [0.1, 0.3, 0.7, 0.8, 0.9],
            "confidence": [0.9, 0.7, 0.7, 0.8, 0.9],
            "quality_score": [80.0] * 5,
            "quality_gate": ["pass"] * 5,
        })
        result = analyze_errors_by_dimension(frame, "dr_grade", threshold=0.5)
        self.assertEqual(len(result), 5)
        for i, item in enumerate(result):
            self.assertEqual(item["dr_grade"], i)
            self.assertEqual(item["n"], 1)

    def test_errors_by_quality_separates_gradable_ungradable(self) -> None:
        frame = pd.DataFrame({
            "partition": ["training", "training"],
            "patient_id": ["1", "2"],
            "image_id": ["1_l1", "2_l1"],
            "eye": ["l", "l"],
            "dr_grade": [0, 0],
            "referable_dr": [0, 0],
            "quality_good": [1, 0],
            "probability_referable_dr": [0.1, 0.2],
            "confidence": [0.9, 0.8],
            "quality_score": [80.0, 40.0],
            "quality_gate": ["pass", "fail"],
        })
        result = analyze_errors_by_dimension(frame, "official_quality", threshold=0.5)
        self.assertEqual(len(result), 2)
        quality_values = {item["official_quality_good"] for item in result}
        self.assertEqual(quality_values, {0, 1})

    def test_errors_by_partition_separates_training_validation(self) -> None:
        frame = pd.DataFrame({
            "partition": ["training", "validation"],
            "patient_id": ["1", "2"],
            "image_id": ["1_l1", "2_l1"],
            "eye": ["l", "l"],
            "dr_grade": [0, 0],
            "referable_dr": [0, 0],
            "quality_good": [1, 1],
            "probability_referable_dr": [0.1, 0.2],
            "confidence": [0.9, 0.8],
            "quality_score": [80.0, 80.0],
            "quality_gate": ["pass", "pass"],
        })
        result = analyze_errors_by_dimension(frame, "partition", threshold=0.5)
        self.assertEqual(len(result), 2)
        partitions = {item["partition"] for item in result}
        self.assertEqual(partitions, {"training", "validation"})


class FailureAnalysisCalibrationTests(unittest.TestCase):
    def test_calibration_calculates_ece_and_brier(self) -> None:
        frame = pd.DataFrame({
            "partition": ["training", "training", "training"],
            "patient_id": ["1", "2", "3"],
            "image_id": ["1_l1", "2_l1", "3_l1"],
            "eye": ["l", "l", "l"],
            "dr_grade": [0, 0, 1],
            "referable_dr": [0, 0, 1],
            "quality_good": [1, 1, 1],
            "probability_referable_dr": [0.1, 0.2, 0.8],
            "confidence": [0.9, 0.8, 0.8],
            "quality_score": [80.0, 80.0, 80.0],
            "quality_gate": ["pass", "pass", "pass"],
        })
        result = analyze_calibration(frame, bins=10)
        self.assertEqual(result["n"], 3)
        self.assertIsNotNone(result["ece"])
        self.assertIsNotNone(result["brier"])
        self.assertEqual(len(result["bins"]), 10)

    def test_calibration_bins_are_complete(self) -> None:
        frame = pd.DataFrame({
            "partition": ["training"] * 3,
            "patient_id": ["1", "2", "3"],
            "image_id": ["1_l1", "2_l1", "3_l1"],
            "eye": ["l", "l", "l"],
            "dr_grade": [0, 0, 1],
            "referable_dr": [0, 0, 1],
            "quality_good": [1, 1, 1],
            "probability_referable_dr": [0.1, 0.2, 0.8],
            "confidence": [0.9, 0.8, 0.8],
            "quality_score": [80.0, 80.0, 80.0],
            "quality_gate": ["pass", "pass", "pass"],
        })
        result = analyze_calibration(frame, bins=5)
        self.assertEqual(len(result["bins"]), 5)
        for bin_result in result["bins"]:
            self.assertIn("bin", bin_result)
            self.assertIn("range", bin_result)
            self.assertIn("n", bin_result)


class FailureAnalysisEyeDisagreementTests(unittest.TestCase):
    def test_eye_disagreement_detects_mismatched_predictions(self) -> None:
        frame = pd.DataFrame({
            "partition": ["training", "training"],
            "patient_id": ["1", "1"],
            "image_id": ["1_l1", "1_l2"],
            "eye": ["l", "l"],
            "dr_grade": [0, 0],
            "referable_dr": [0, 0],
            "quality_good": [1, 1],
            "probability_referable_dr": [0.4, 0.6],  # Different predictions at threshold 0.5
            "confidence": [0.6, 0.6],
            "quality_score": [80.0, 80.0],
            "quality_gate": ["pass", "pass"],
        })
        result = analyze_eye_pair_disagreements(frame, threshold=0.5)
        self.assertEqual(result["total_pairs"], 1)
        self.assertEqual(result["disagreement_count"], 1)

    def test_eye_disagreement_no_disagreement_when_same_predictions(self) -> None:
        frame = pd.DataFrame({
            "partition": ["training", "training"],
            "patient_id": ["1", "1"],
            "image_id": ["1_l1", "1_l2"],
            "eye": ["l", "l"],
            "dr_grade": [0, 0],
            "referable_dr": [0, 0],
            "quality_good": [1, 1],
            "probability_referable_dr": [0.4, 0.3],  # Same predictions at threshold 0.5
            "confidence": [0.6, 0.7],
            "quality_score": [80.0, 80.0],
            "quality_gate": ["pass", "pass"],
        })
        result = analyze_eye_pair_disagreements(frame, threshold=0.5)
        self.assertEqual(result["total_pairs"], 1)
        self.assertEqual(result["disagreement_count"], 0)

    def test_eye_disagreement_groups_by_partition(self) -> None:
        frame = pd.DataFrame({
            "partition": ["training", "validation", "training", "validation"],
            "patient_id": ["1", "2", "1", "2"],
            "image_id": ["1_l1", "2_l1", "1_l2", "2_l2"],
            "eye": ["l", "l", "l", "l"],
            "dr_grade": [0, 0, 0, 0],
            "referable_dr": [0, 0, 0, 0],
            "quality_good": [1, 1, 1, 1],
            "probability_referable_dr": [0.4, 0.6, 0.3, 0.7],
            "confidence": [0.6, 0.6, 0.7, 0.7],
            "quality_score": [80.0, 80.0, 80.0, 80.0],
            "quality_gate": ["pass", "pass", "pass", "pass"],
        })
        result = analyze_eye_pair_disagreements(frame, threshold=0.5)
        self.assertEqual(len(result["by_partition"]), 2)
        partitions = {item["partition"] for item in result["by_partition"]}
        self.assertEqual(partitions, {"training", "validation"})


class FailureAnalysisBootstrapTests(unittest.TestCase):
    def test_bootstrap_requires_at_least_two_clusters(self) -> None:
        frame = pd.DataFrame({
            "partition": ["training"],
            "patient_id": ["1"],  # Only one patient
            "image_id": ["1_l1"],
            "eye": ["l"],
            "dr_grade": [0],
            "referable_dr": [0],
            "quality_good": [1],
            "probability_referable_dr": [0.1],
            "confidence": [0.9],
            "quality_score": [80.0],
            "quality_gate": ["pass"],
        })
        result = calculate_bootstrap_intervals(frame, samples=100, seed=42, threshold=0.5)
        self.assertEqual(result["cluster_count"], 1)
        self.assertIsNone(result["intervals"])
        self.assertIn("At least two patient clusters", result["reason"])

    def test_bootstrap_calculates_intervals_with_multiple_clusters(self) -> None:
        frame = pd.DataFrame({
            "partition": ["training", "training", "training", "training"],
            "patient_id": ["1", "1", "2", "2"],  # Two patients
            "image_id": ["1_l1", "1_l2", "2_l1", "2_l2"],
            "eye": ["l", "l", "l", "l"],
            "dr_grade": [0, 0, 1, 1],
            "referable_dr": [0, 0, 1, 1],
            "quality_good": [1, 1, 1, 1],
            "probability_referable_dr": [0.1, 0.2, 0.7, 0.8],
            "confidence": [0.9, 0.8, 0.7, 0.8],
            "quality_score": [80.0, 80.0, 80.0, 80.0],
            "quality_gate": ["pass", "pass", "pass", "pass"],
        })
        result = calculate_bootstrap_intervals(frame, samples=100, seed=42, threshold=0.5)
        self.assertEqual(result["cluster_count"], 2)
        self.assertIsNotNone(result["intervals"])
        self.assertIn("95_ci", result["intervals"]["intervals"]["auroc"])


class FailureAnalysisReportTests(unittest.TestCase):
    def test_bulgarian_report_generates_markdown(self) -> None:
        result = {
            "generated_at_utc": "2026-10-27T10:00:00Z",
            "input_file": "test.csv",
            "total_images": 100,
            "total_patients": 20,
            "dr_eligible_images": 90,
            "validation": {
                "duplicate_image_ids": [],
                "invalid_dr_grades": [],
                "invalid_quality_gates": [],
                "missing_referable_dr": 0,
            },
            "errors_by_dr_grade": [
                {"dr_grade": 0, "n": 30, "error_rate": 0.1, "mean_probability": 0.2},
            ],
            "errors_by_quality": [
                {"official_quality_good": 1, "n": 50, "error_rate": 0.15, "mean_probability": 0.3},
            ],
            "errors_by_partition": [
                {"partition": "training", "n": 60, "error_rate": 0.12, "mean_probability": 0.25},
            ],
            "errors_by_gate": [
                {"quality_gate": "pass", "n": 40, "error_rate": 0.1, "mean_probability": 0.2},
            ],
            "calibration": {
                "n": 90,
                "ece": 0.15,
                "brier": 0.2,
            },
            "eye_disagreements": {
                "total_pairs": 10,
                "disagreement_count": 2,
                "by_partition": [
                    {"partition": "training", "total_pairs": 6, "disagreement_pairs": 1, "disagreement_rate": 0.1667},
                ],
            },
            "bootstrap_intervals": {
                "cluster_count": 20,
                "intervals": {
                    "intervals": {
                        "auroc": {"95_ci": [0.4, 0.5]},
                    },
                },
            },
            "output_json": "outputs/failure_analysis.json",
            "output_csv": "outputs/disagreements.csv",
        }
        report = generate_bulgarian_report(result)
        self.assertIn("# RetinaTrust v0.6", report)
        self.assertIn("Анализ на грешките", report)
        self.assertIn("Калибрация", report)
        self.assertIn("0.15", report)  # ECE value
        self.assertIn("0.2", report)  # Brier score


if __name__ == "__main__":
    unittest.main()
