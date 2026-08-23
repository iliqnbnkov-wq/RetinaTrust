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

from retina_poc.external_validation import (  # noqa: E402
    aggregate_eye_predictions,
    aggregate_patient_predictions,
    binary_probability_metrics,
    cluster_bootstrap_intervals,
    quality_gate_metrics,
    read_deepdrid_csv,
    sha256_file,
    verify_protocol_lock,
)


HEADER = (
    "patient_id,image_id,image_path,Overall quality,left_eye_DR_Level,"
    "right_eye_DR_Level,patient_DR_Level,Clarity,Field definition,Artifact\n"
)


class DeepDRiDParsingTests(unittest.TestCase):
    def test_official_schema_maps_eye_grade_quality_and_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split = Path(directory) / "regular-fundus-training"
            image_dir = split / "Images/7"
            image_dir.mkdir(parents=True)
            image_path = image_dir / "7_l1.jpg"
            image_path.touch()
            csv_path = split / "regular-fundus-training.csv"
            csv_path.write_text(
                HEADER + "7,7_l1,\\regular-fundus-training\\7\\7_l1.jpg,1,2,,2,8,10,0\n",
                encoding="utf-8",
            )

            records = read_deepdrid_csv(csv_path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].eye, "l")
            self.assertEqual(records[0].dr_grade, 2)
            self.assertEqual(records[0].referable_dr, 1)
            self.assertEqual(records[0].quality_good, 1)
            self.assertEqual(records[0].image_path, image_path)

    def test_grade_five_is_quality_only_not_binary_dr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split = Path(directory) / "regular-fundus-validation"
            split.mkdir(parents=True)
            csv_path = split / "regular-fundus-validation.csv"
            csv_path.write_text(
                HEADER + "8,8_r2,\\regular-fundus-validation\\8\\8_r2.jpg,0,,5,5,1,1,10\n",
                encoding="utf-8",
            )
            record = read_deepdrid_csv(csv_path, require_images=False)[0]
            self.assertIsNone(record.referable_dr)

    def test_side_and_grade_columns_must_agree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split = Path(directory) / "regular-fundus-training"
            split.mkdir(parents=True)
            csv_path = split / "regular-fundus-training.csv"
            csv_path.write_text(
                HEADER + "9,9_l1,\\regular-fundus-training\\9\\9_l1.jpg,1,,2,2,8,8,0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "inconsistent eye-grade"):
                read_deepdrid_csv(csv_path, require_images=False)


class ExternalMetricsTests(unittest.TestCase):
    def test_binary_metrics_use_explicit_confusion_semantics(self) -> None:
        metrics = binary_probability_metrics(
            [0, 0, 1, 1], [0.1, 0.8, 0.4, 0.9], threshold=0.5
        )
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(metrics["sensitivity"], 0.5)
        self.assertEqual(metrics["specificity"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)

    def test_quality_gate_reports_under_and_over_rejection(self) -> None:
        result = quality_gate_metrics(
            quality_good=[1, 1, 0, 0],
            gates=["pass", "review", "pass", "fail"],
        )
        primary = result["intervention_review_or_fail"]
        self.assertEqual(primary["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(primary["under_rejection_rate"], 0.5)
        self.assertEqual(primary["over_rejection_rate"], 0.5)

    def test_patient_cluster_bootstrap_is_reproducible(self) -> None:
        y = [0, 0, 1, 1, 0, 0, 1, 1]
        probability = [0.1, 0.2, 0.7, 0.8, 0.3, 0.4, 0.6, 0.9]
        patients = ["a", "a", "b", "b", "c", "c", "d", "d"]
        first = cluster_bootstrap_intervals(
            y, probability, patients, samples=100, seed=42
        )
        second = cluster_bootstrap_intervals(
            y, probability, patients, samples=100, seed=42
        )
        self.assertEqual(first, second)
        self.assertEqual(first["cluster_count"], 4)
        self.assertGreater(first["intervals"]["auroc"]["valid_replicates"], 0)

    def test_eye_and_patient_aggregation_are_secondary_max_rules(self) -> None:
        frame = pd.DataFrame(
            {
                "patient_id": ["1", "1", "1", "1", "2", "2"],
                "eye": ["l", "l", "r", "r", "l", "l"],
                "dr_grade": [0, 0, 2, 2, 1, 1],
                "patient_dr_grade": [2, 2, 2, 2, 1, 1],
                "probability_referable_dr": [0.1, 0.3, 0.6, 0.8, 0.2, 0.4],
                "quality_gate": ["pass", "review", "pass", "pass", "fail", "review"],
            }
        )
        eye = aggregate_eye_predictions(frame)
        patient = aggregate_patient_predictions(frame)
        patient_one = patient[patient["patient_id"] == "1"].iloc[0]
        left_one = eye[(eye["patient_id"] == "1") & (eye["eye"] == "l")].iloc[0]
        self.assertAlmostEqual(float(left_one["probability_referable_dr"]), 0.3)
        self.assertAlmostEqual(float(patient_one["probability_referable_dr"]), 0.8)
        self.assertEqual(int(patient_one["referable_dr"]), 1)

    def test_protocol_lock_rejects_an_unfrozen_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen_model = root / "frozen.joblib"
            other_model = root / "other.joblib"
            scientific_code = root / "pipeline.py"
            frozen_model.write_bytes(b"frozen")
            other_model.write_bytes(b"other")
            scientific_code.write_text("VALUE = 1\n", encoding="utf-8")
            lock = {
                "schema_version": 1,
                "status": "frozen_before_external_results",
                "model_path": "frozen.joblib",
                "thresholds": {"classification": 0.5, "confidence_floor": 0.85},
                "files": [
                    {"path": "frozen.joblib", "sha256": sha256_file(frozen_model)},
                    {"path": "pipeline.py", "sha256": sha256_file(scientific_code)},
                ],
            }
            lock_path = root / "lock.json"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            bundle = {
                "classification_threshold": 0.5,
                "review_confidence_threshold": 0.85,
            }
            with self.assertRaisesRegex(ValueError, "not the model frozen"):
                verify_protocol_lock(lock_path, root, bundle, other_model)


if __name__ == "__main__":
    unittest.main()
