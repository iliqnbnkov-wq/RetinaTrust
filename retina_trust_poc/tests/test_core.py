from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retina_poc.core import (  # noqa: E402
    CANONICAL_WORKING_SIZE,
    apply_degradation,
    canonicalize_image,
    extract_features,
    load_rgb,
)
from retina_poc.modeling import load_bundle, predict_case  # noqa: E402


class RetinaTrustCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image = load_rgb(ROOT / "demo/IDRiD_001_demo.jpg")
        cls.bundle = load_bundle(ROOT / "artifacts/retina_baseline.joblib")

    def test_feature_schema_matches_model(self) -> None:
        vector, names, metrics = extract_features(self.image)
        self.assertEqual(len(vector), 178)
        self.assertEqual(names, self.bundle["feature_names"])
        self.assertGreater(metrics["field_coverage"], 0.5)

    def test_clean_demo_is_not_rejected_for_geometry(self) -> None:
        result = predict_case(self.image, self.bundle)
        self.assertEqual(result["quality"]["gate"], "pass")
        self.assertGreaterEqual(result["quality"]["components"]["field_coverage"], 95)

    def test_extreme_combined_perturbation_is_rejected(self) -> None:
        degraded = apply_degradation(self.image, "combined", 3)
        result = predict_case(degraded, self.bundle)
        self.assertEqual(result["quality"]["gate"], "fail")
        self.assertEqual(result["decision"]["code"], "recapture")

    def test_invalid_level_fails_explicitly(self) -> None:
        with self.assertRaises(ValueError):
            apply_degradation(self.image, "blur", 4)

    def test_clean_and_degraded_paths_share_canonical_geometry(self) -> None:
        clean = apply_degradation(self.image, "none", 1)
        self.assertEqual(max(clean.size), CANONICAL_WORKING_SIZE)
        for kind in ("underexposure", "overexposure", "gamma", "color_shift", "blur", "jpeg", "combined"):
            with self.subTest(kind=kind):
                self.assertEqual(apply_degradation(self.image, kind, 2).size, clean.size)

    def test_canonicalization_is_idempotent_at_working_resolution(self) -> None:
        first = canonicalize_image(self.image)
        second = canonicalize_image(first)
        self.assertEqual(first.size, second.size)
        self.assertEqual(first.tobytes(), second.tobytes())


if __name__ == "__main__":
    unittest.main()
