from __future__ import annotations

import base64
import io
import json
import socket
import sys
import threading
import time
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

from PIL import Image as PILImage


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import ARTIFACT, RetinaHandler  # noqa: E402
from retina_poc.core import image_to_data_url, load_rgb  # noqa: E402
from retina_poc.modeling import load_bundle  # noqa: E402
from retina_poc.validation import MAX_REQUEST_BYTES  # noqa: E402


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.bind(("127.0.0.1", 0))
        return int(candidate.getsockname()[1])


class TestServerIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.host = "127.0.0.1"
        cls.port = find_free_port()
        RetinaHandler.bundle = load_bundle(ARTIFACT)
        cls.server = ThreadingHTTPServer((cls.host, cls.port), RetinaHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.1)

        cls.demo_image = load_rgb(ROOT / "demo/IDRiD_001_demo.jpg")
        cls.sample_data_url = image_to_data_url(cls.demo_image)
        cls.original_data_url = (
            "data:image/jpeg;base64,"
            + base64.b64encode((ROOT / "demo/IDRiD_001_demo.jpg").read_bytes()).decode("ascii")
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(
        self,
        method: str,
        path: str,
        body: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        connection = HTTPConnection(self.host, self.port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def analyze_payload(self, **overrides: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "image_data_url": self.sample_data_url,
            "degradation": "none",
            "level": 1,
            "normalize": False,
        }
        payload.update(overrides)
        return payload

    def post_json(self, payload: object) -> tuple[int, dict[str, str], dict[str, object]]:
        body = json.dumps(payload)
        status, headers, raw = self.request(
            "POST",
            "/api/analyze",
            body,
            {"Content-Type": "application/json"},
        )
        return status, headers, json.loads(raw.decode("utf-8"))

    def test_health_endpoint(self) -> None:
        status, headers, raw = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["status"], "ok")
        self.assertEqual(headers["X-Frame-Options"], "DENY")
        self.assertIn("default-src 'self'", headers["Content-Security-Policy"])

    def test_status_exposes_model_metrics_and_robustness(self) -> None:
        status, _, raw = self.request("GET", "/api/status")
        payload = json.loads(raw)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["model"]["version"], "0.3.0")
        self.assertEqual(payload["metrics"]["n"], 103)
        self.assertEqual(
            payload["presentation"]["technical_confidence_definition"],
            "max(p, 1 - p)",
        )
        self.assertIn("underexposure", {row["condition"] for row in payload["robustness"]})
        underexposure = next(
            row for row in payload["robustness"] if row["condition"] == "underexposure"
        )
        self.assertLess(underexposure["delta_auroc_ci"][1], 0.0)
        clean = next(row for row in payload["robustness"] if row["condition"] == "none")
        self.assertAlmostEqual(clean["quality_intervention_rate"], 0.252, places=3)

    def test_sample_endpoint_returns_jpeg_data_url(self) -> None:
        status, _, raw = self.request("GET", "/api/sample")
        payload = json.loads(raw)
        self.assertEqual(status, 200)
        self.assertTrue(payload["image_data_url"].startswith("data:image/jpeg;base64,"))

    def test_sample_roundtrip_scientific_regression(self) -> None:
        status, _, payload = self.post_json(self.analyze_payload())
        self.assertEqual(status, 200)
        self.assertAlmostEqual(payload["probability_referable_dr"], 0.9521, places=4)
        self.assertAlmostEqual(payload["quality"]["score"], 96.7, places=1)
        self.assertEqual(payload["quality"]["gate"], "pass")
        self.assertEqual(payload["decision"]["code"], "provisional_positive")
        self.assertNotIn("confidence", payload)
        self.assertNotIn("uncertainty", payload)
        self.assertAlmostEqual(payload["technical_confidence"]["value"], 0.9521, places=4)
        self.assertEqual(
            payload["technical_confidence"]["definition"],
            "max(p, 1 - p)",
        )
        self.assertIn("не е отделна оценка", payload["technical_confidence"]["limitation"])
        self.assertEqual(payload["thresholds"]["technical_confidence_floor"], 0.85)
        self.assertNotIn("confidence", payload["thresholds"])

    def test_interface_uses_explicit_non_clinical_semantics(self) -> None:
        status, _, raw = self.request("GET", "/")
        html = raw.decode("utf-8")
        self.assertEqual(status, 200)
        self.assertIn("Техническа увереност", html)
        self.assertIn("Експерименталното правило предлага", html)
        self.assertNotIn("Увереност на модела", html)
        self.assertNotIn("Системата изисква повторно заснемане", html)

    def test_original_demo_bytes_scientific_regression(self) -> None:
        status, _, payload = self.post_json(
            self.analyze_payload(image_data_url=self.original_data_url)
        )
        self.assertEqual(status, 200)
        # Native JPEG decoders can differ by one least-significant feature step
        # across operating systems. Keep the regression narrow while allowing
        # the observed Windows/Linux probability variation.
        self.assertAlmostEqual(
            payload["probability_referable_dr"],
            0.9450,
            delta=0.0002,
        )
        self.assertAlmostEqual(payload["quality"]["score"], 97.0, places=1)

    def test_severe_combined_input_is_blocked_for_recapture(self) -> None:
        status, _, payload = self.post_json(
            self.analyze_payload(degradation="combined", level=3)
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["quality"]["gate"], "fail")
        self.assertEqual(payload["decision"]["code"], "recapture")

    def test_invalid_json_is_client_error(self) -> None:
        status, _, raw = self.request(
            "POST",
            "/api/analyze",
            "not-json",
            {"Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(raw))

    def test_non_object_json_is_rejected(self) -> None:
        status, _, payload = self.post_json(["not", "an", "object"])
        self.assertEqual(status, 400)
        self.assertIn("JSON обект", payload["error"])

    def test_content_type_is_required(self) -> None:
        body = json.dumps(self.analyze_payload())
        status, _, raw = self.request("POST", "/api/analyze", body)
        self.assertEqual(status, 415)
        self.assertIn("application/json", json.loads(raw)["error"])

    def test_wrong_content_type_is_rejected(self) -> None:
        body = json.dumps(self.analyze_payload())
        status, _, raw = self.request(
            "POST", "/api/analyze", body, {"Content-Type": "text/plain"}
        )
        self.assertEqual(status, 415)
        self.assertIn("application/json", json.loads(raw)["error"])

    def test_missing_image_field_is_rejected(self) -> None:
        status, _, payload = self.post_json(
            {"degradation": "none", "level": 1, "normalize": False}
        )
        self.assertEqual(status, 400)
        self.assertIn("image_data_url", payload["error"])

    def test_invalid_base64_is_rejected(self) -> None:
        status, _, payload = self.post_json(
            self.analyze_payload(image_data_url="data:image/jpeg;base64,not!!!base64")
        )
        self.assertEqual(status, 400)
        self.assertIn("Base64", payload["error"])

    def test_unsupported_mime_type_is_rejected(self) -> None:
        status, _, payload = self.post_json(
            self.analyze_payload(
                image_data_url="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="
            )
        )
        self.assertEqual(status, 400)
        self.assertIn("JPEG", payload["error"])

    def test_missing_base64_marker_is_rejected(self) -> None:
        encoded = self.sample_data_url.split(",", 1)[1]
        status, _, payload = self.post_json(
            self.analyze_payload(image_data_url=f"data:image/jpeg,{encoded}")
        )
        self.assertEqual(status, 400)

    def test_declared_mime_must_match_decoded_format(self) -> None:
        buffer = io.BytesIO()
        self.demo_image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        status, _, payload = self.post_json(
            self.analyze_payload(image_data_url=f"data:image/jpeg;base64,{encoded}")
        )
        self.assertEqual(status, 400)
        self.assertIn("не съответства", payload["error"])

    def test_tiny_image_is_rejected(self) -> None:
        image = PILImage.new("RGB", (64, 64), "red")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        status, _, payload = self.post_json(self.analyze_payload(image_data_url=data_url))
        self.assertEqual(status, 400)
        self.assertIn("256", payload["error"])

    def test_excessive_pixel_count_is_rejected_before_decode(self) -> None:
        image = PILImage.new("L", (7100, 7100), 20)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        status, _, payload = self.post_json(self.analyze_payload(image_data_url=data_url))
        self.assertEqual(status, 400)
        self.assertIn("пиксели", payload["error"])

    def test_invalid_degradation_is_rejected(self) -> None:
        status, _, payload = self.post_json(
            self.analyze_payload(degradation="invented-corruption")
        )
        self.assertEqual(status, 400)
        self.assertIn("Неподдържано", payload["error"])

    def test_degradation_must_be_string(self) -> None:
        status, _, _ = self.post_json(self.analyze_payload(degradation=123))
        self.assertEqual(status, 400)

    def test_level_outside_fixed_matrix_is_rejected(self) -> None:
        status, _, payload = self.post_json(self.analyze_payload(level=4))
        self.assertEqual(status, 400)
        self.assertIn("1, 2 или 3", payload["error"])

    def test_boolean_level_is_not_treated_as_integer(self) -> None:
        status, _, _ = self.post_json(self.analyze_payload(level=True))
        self.assertEqual(status, 400)

    def test_float_level_is_rejected(self) -> None:
        status, _, _ = self.post_json(self.analyze_payload(level=1.0))
        self.assertEqual(status, 400)

    def test_normalize_must_be_boolean(self) -> None:
        status, _, payload = self.post_json(self.analyze_payload(normalize="false"))
        self.assertEqual(status, 400)
        self.assertIn("true или false", payload["error"])

    def test_oversized_content_length_is_rejected_before_body_read(self) -> None:
        connection = HTTPConnection(self.host, self.port, timeout=5)
        try:
            connection.putrequest("POST", "/api/analyze")
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", str(MAX_REQUEST_BYTES + 1))
            connection.endheaders()
            response = connection.getresponse()
            payload = json.loads(response.read())
            self.assertEqual(response.status, 413)
            self.assertIn("20 MB", payload["error"])
        finally:
            connection.close()

    def test_static_path_traversal_is_forbidden(self) -> None:
        status, _, _ = self.request("GET", "/../../../etc/passwd")
        self.assertEqual(status, 403)


if __name__ == "__main__":
    unittest.main()
