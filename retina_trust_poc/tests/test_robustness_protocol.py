from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_robustness import paired_comparison  # noqa: E402


class RobustnessProtocolTests(unittest.TestCase):
    def test_identical_probabilities_have_zero_paired_differences(self) -> None:
        y = np.asarray([0, 0, 1, 1, 0, 1])
        probability = np.asarray([0.1, 0.3, 0.7, 0.9, 0.2, 0.8])
        result = paired_comparison(y, probability, probability.copy(), samples=100)
        for key in (
            "delta_auroc",
            "delta_macro_f1",
            "delta_balanced_accuracy",
            "prediction_flip_rate",
            "mean_absolute_probability_shift",
            "mean_signed_probability_shift",
        ):
            self.assertEqual(result[key], 0.0)
            self.assertEqual(result["paired_bootstrap_95_ci"][key], [0.0, 0.0])

    def test_flip_rate_is_case_paired(self) -> None:
        y = np.asarray([0, 0, 1, 1])
        clean = np.asarray([0.1, 0.4, 0.6, 0.9])
        changed = np.asarray([0.8, 0.4, 0.3, 0.9])
        result = paired_comparison(y, clean, changed, samples=100)
        self.assertEqual(result["prediction_flip_rate"], 0.5)
        self.assertGreater(result["mean_absolute_probability_shift"], 0.0)


if __name__ == "__main__":
    unittest.main()
