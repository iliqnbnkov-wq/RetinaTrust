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
    calculate_overall_metrics,
    generate_bulgarian_report,
    validate_predictions_csv,
)


class FailureAnalysisValidationTests(unittest.TestCase):
    def test_validate_rejects_duplicate_image_ids(self) -> None:
        frame = pd.DataFrame({
            "partition": ["regular-fundus-training", "regular-fundus-training"],
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
        with self.assertRaisesRegex(ValueError, "duplicate image_ids"):
            validate_predictions_csv(frame, threshold=0.5)

    def test_validate_rejects_invalid_dr_grades(self) -> None:
        frame = pd.DataFrame({
            "partition": ["regular-fundus-training"],
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
        with self.assertRaisesRegex(ValueError, "invalid DR grades"):
            validate_predictions_csv(frame, threshold=0.5)

    def test_validate_rejects_invalid_quality_gates(self) -> None:
        frame = pd.DataFrame({
            "partition": ["regular-fundus-training"],
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
        with self.assertRaisesRegex(ValueError, "invalid quality gates"):
            validate_predictions_csv(frame, threshold=0.5)

    def test_validate_rejects_missing_referable_dr(self) -> None:
        frame = pd.DataFrame({
            "partition": ["regular-fundus-training"],
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
        with self.assertRaisesRegex(ValueError, "missing referable_dr"):
            validate_predictions_csv(frame, threshold=0.5)

    def test_validate_rejects_nonfinite_probabilities(self) -> None:
        frame = pd.DataFrame({
            "partition": ["regular-fundus-training"],
            "patient_id": ["1"],
            "image_id": ["1_l1"],
            "eye": ["l"],
            "dr_grade": [0],
            "referable_dr": [0],
            "quality_good": [1],
            "probability_referable_dr": [np.nan],  # Nonfinite
            "confidence": [0.9],
            "quality_score": [80.0],
            "quality_gate": ["pass"],
        })
        with self.assertRaisesRegex(ValueError, "nonfinite or out-of-range probabilities"):
            validate_predictions_csv(frame, threshold=0.5)

    def test_validate_rejects_invalid_threshold(self) -> None:
        frame = pd.DataFrame({
            "partition": ["regular-fundus-training"],
            "patient_id": ["1"],
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
        with self.assertRaisesRegex(ValueError, "Classification threshold must be between 0 and 1"):
            validate_predictions_csv(frame, threshold=1.5)

    def test_validate_rejects_inconsistent_dr_labels(self) -> None:
        frame = pd.DataFrame({
            "partition": ["regular-fundus-training"],
            "patient_id": ["1"],
            "image_id": ["1_l1"],
            "eye": ["l"],
            "dr_grade": [2],  # Should be referable (1)
            "referable_dr": [0],  # But is marked as non-referable
            "quality_good": [1],
            "probability_referable_dr": [0.1],
            "confidence": [0.9],
            "quality_score": [80.0],
            "quality_gate": ["pass"],
        })
        with self.assertRaisesRegex(ValueError, "inconsistent DR grades and binary labels"):
            validate_predictions_csv(frame, threshold=0.5)

    def test_validate_accepts_ungradable_dr_grade_five(self) -> None:
        frame = pd.DataFrame({
            "partition": ["regular-fundus-training"],
            "patient_id": ["1"],
            "image_id": ["1_l1"],
            "eye": ["l"],
            "dr_grade": [5],  # UNGRADABLE_DR_GRADE is valid
            "referable_dr": [None],  # Can be None for ungradable
            "quality_good": [0],
            "probability_referable_dr": [0.1],
            "confidence": [0.9],
            "quality_score": [40.0],
            "quality_gate": ["fail"],
        })
        # Should not raise
        result = validate_predictions_csv(frame, threshold=0.5)
        self.assertEqual(result["total_rows"], 1)


class FailureAnalysisErrorTests(unittest.TestCase):
    def test_errors_by_dr_grade_returns_all_grades(self) -> None:
        frame = pd.DataFrame({
            "partition": ["regular-fundus-training"] * 5,
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
            "partition": ["regular-fundus-training", "regular-fundus-training"],
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

    def test_errors_by_partition_uses_actual_deepdrid_names(self) -> None:
        frame = pd.DataFrame({
            "partition": ["regular-fundus-training", "regular-fundus-validation"],
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
        self.assertEqual(partitions, {"regular-fundus-training", "regular-fundus-validation"})


class FailureAnalysisCalibrationTests(unittest.TestCase):
    def test_calibration_calculates_ece_and_brier(self) -> None:
        frame = pd.DataFrame({
            "partition": ["regular-fundus-training", "regular-fundus-training", "regular-fundus-training"],
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

    def test_calibration_uses_correct_terminology(self) -> None:
        frame = pd.DataFrame({
            "partition": ["regular-fundus-training"] * 3,
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
        # Check that the correct terminology is used
        for bin_result in result["bins"]:
            if bin_result["n"] > 0:
                self.assertIn("mean_predicted_positive_probability", bin_result)
                self.assertIn("observed_positive_rate", bin_result)


class FailureAnalysisEyeDisagreementTests(unittest.TestCase):
    def test_eye_disagreement_detects_mismatched_predictions(self) -> None:
        frame = pd.DataFrame({
            "partition": ["regular-fundus-training", "regular-fundus-training"],
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

    def test_eye_disagreement_uses_actual_partition_names(self) -> None:
        frame = pd.DataFrame({
            "partition": ["regular-fundus-training", "regular-fundus-validation", "regular-fundus-training", "regular-fundus-validation"],
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
        self.assertEqual(partitions, {"regular-fundus-training", "regular-fundus-validation"})


class FailureAnalysisBootstrapTests(unittest.TestCase):
    def test_bootstrap_requires_at_least_two_clusters(self) -> None:
        frame = pd.DataFrame({
            "partition": ["regular-fundus-training"],
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
            "partition": ["regular-fundus-training", "regular-fundus-training", "regular-fundus-training", "regular-fundus-training"],
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


class FailureAnalysisOverallMetricsTests(unittest.TestCase):
    def test_overall_metrics_calculates_key_metrics(self) -> None:
        frame = pd.DataFrame({
            "partition": ["regular-fundus-training", "regular-fundus-training", "regular-fundus-training", "regular-fundus-training"],
            "patient_id": ["1", "2", "3", "4"],
            "image_id": ["1_l1", "2_l1", "3_l1", "4_l1"],
            "eye": ["l", "l", "l", "l"],
            "dr_grade": [0, 0, 1, 1],
            "referable_dr": [0, 0, 1, 1],
            "quality_good": [1, 1, 1, 1],
            "probability_referable_dr": [0.1, 0.2, 0.7, 0.8],
            "confidence": [0.9, 0.8, 0.7, 0.8],
            "quality_score": [80.0, 80.0, 80.0, 80.0],
            "quality_gate": ["pass", "pass", "pass", "pass"],
        })
        result = calculate_overall_metrics(frame, threshold=0.5)
        self.assertEqual(result["n"], 4)
        self.assertIsNotNone(result["metrics"])
        metrics = result["metrics"]
        self.assertIn("auroc", metrics)
        self.assertIn("accuracy", metrics)
        self.assertIn("sensitivity", metrics)
        self.assertIn("specificity", metrics)
        self.assertIn("brier", metrics)
        self.assertIn("ece", metrics)
        self.assertIn("confusion_matrix", metrics)


class FailureAnalysisReportTests(unittest.TestCase):
    def test_bulgarian_report_generates_markdown(self) -> None:
        result = {
            "generated_at_utc": "2026-10-27T10:00:00Z",
            "input_file": "test.csv",
            "input_sha256": "abc123",
            "total_images": 100,
            "total_patients": 20,
            "dr_eligible_images": 90,
            "overall_metrics": {
                "metrics": {
                    "auroc": 0.4620,
                    "accuracy": 0.4462,
                    "sensitivity": 0.7235,
                    "specificity": 0.2290,
                    "balanced_accuracy": 0.4762,
                    "macro_f1": 0.45,
                    "brier": 0.4390,
                    "nll": 0.65,
                    "ece": 0.4100,
                    "confusion_matrix": [[204, 687], [193, 505]],
                }
            },
            "errors_by_dr_grade": [
                {"dr_grade": 0, "n": 30, "error_rate": 0.1, "mean_probability": 0.2},
            ],
            "errors_by_quality": [
                {"official_quality_good": 1, "n": 50, "error_rate": 0.15, "mean_probability": 0.3},
            ],
            "errors_by_partition": [
                {"partition": "regular-fundus-training", "n": 60, "error_rate": 0.12, "mean_probability": 0.25},
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
                    {"partition": "regular-fundus-training", "total_pairs": 6, "disagreement_pairs": 1, "disagreement_rate": 0.1667},
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
        self.assertIn("0.4620", report)  # AUROC value
        self.assertIn("0.4390", report)  # Brier score
        self.assertIn("regular-fundus-training", report)  # Actual partition name


if __name__ == "__main__":
    unittest.main()
