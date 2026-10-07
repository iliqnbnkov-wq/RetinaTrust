#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from retina_poc.core import (
    SUPPORTED_DEGRADATIONS,
    apply_degradation,
    image_to_data_url,
    load_rgb,
    normalize_image,
)
from retina_poc.modeling import load_bundle, make_local_contrast_map, predict_case
from retina_poc.validation import (
    ValidationError,
    parse_and_validate_request,
    parse_content_length,
    validate_content_type,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("retinatrust")


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
ARTIFACT = ROOT / "artifacts/v0.3/retina_baseline.joblib"
DEMO_IMAGE = ROOT / "demo/IDRiD_001_demo.jpg"
ROBUSTNESS_RESULTS = ROOT / "artifacts/v0.3/robustness_results.json"

TECHNICAL_CONFIDENCE_DEFINITION = "max(p, 1 - p)"
TECHNICAL_CONFIDENCE_LIMITATION = (
    "Вероятността на избрания клас не е отделна оценка на епистемична или "
    "предсказвателна несигурност и не открива надеждно всички входове извън "
    "обучаващото разпределение."
)


def compact_metrics(bundle: dict) -> dict:
    metrics = bundle["locked_test_metrics"]
    return {
        key: metrics[key]
        for key in (
            "n",
            "auroc",
            "sensitivity",
            "specificity",
            "f1",
            "macro_f1",
            "balanced_accuracy",
            "brier",
            "nll",
            "ece",
            "confusion_matrix",
            "bootstrap_95_ci",
            "risk_coverage",
            "model_selection",
            "data",
        )
    }


def compact_robustness() -> list[dict]:
    if not ROBUSTNESS_RESULTS.exists():
        return []
    payload = json.loads(ROBUSTNESS_RESULTS.read_text(encoding="utf-8"))
    rows = []
    for row in payload.get("results", []):
        if row.get("condition") == "none" or (
            row.get("level") == 3 and not row.get("normalization", False)
        ):
            paired = row.get("paired_vs_clean", {})
            paired_ci = paired.get("paired_bootstrap_95_ci", {})
            rows.append(
                {
                    "condition": row["condition"],
                    "level": row["level"],
                    "auroc": row["auroc"],
                    "macro_f1": row["macro_f1"],
                    "delta_auroc": paired.get("delta_auroc", 0.0),
                    "delta_auroc_ci": paired_ci.get("delta_auroc", [0.0, 0.0]),
                    "prediction_flip_rate": paired.get("prediction_flip_rate", 0.0),
                    "prediction_flip_rate_ci": paired_ci.get(
                        "prediction_flip_rate", [0.0, 0.0]
                    ),
                    "quality_intervention_rate": None
                    if row.get("quality_fail_rate") is None
                    else row["quality_fail_rate"] + row["quality_review_rate"],
                }
            )
    return rows


def quality_gate_limitation(rows: list[dict] | None = None) -> str:
    rows = compact_robustness() if rows is None else rows
    clean = next((row for row in rows if row["condition"] == "none"), None)
    if clean is None or clean["quality_intervention_rate"] is None:
        evidence = "Няма налична clean оценка на честотата на намеса."
    else:
        rate = float(clean["quality_intervention_rate"])
        evidence = f"То се намесва при {rate:.1%} от чистия IDRiD test split."
    return (
        "Quality gate е ръчно зададено експериментално правило. "
        f"{evidence} Няма клинична валидация."
    )


def present_case_result(result: dict) -> dict:
    """Expose the frozen model output with scientifically explicit semantics.

    The v0.3 model and its scientific code are integrity-locked. This adapter
    changes only the public presentation contract: it removes the ambiguous
    scalar names ``confidence`` and ``uncertainty`` and explains exactly what
    the displayed quantity means.
    """
    presented = dict(result)
    technical_confidence = float(presented.pop("confidence"))
    presented.pop("uncertainty", None)

    thresholds = dict(presented["thresholds"])
    confidence_floor = float(thresholds.pop("confidence"))
    thresholds["technical_confidence_floor"] = confidence_floor
    presented["thresholds"] = thresholds

    decision = dict(presented["decision"])
    if decision["code"] == "recapture":
        decision["reason"] = (
            "Ръчно зададеният quality gate е отбелязал входа като fail. "
            "Това е експериментално правило, а не клинична оценка на годността."
        )
    elif decision["code"] == "review":
        decision["reason"] = (
            "Quality gate е review или max(p, 1 - p) е под фиксирания праг. "
            "Случаят остава за човешки преглед."
        )
    else:
        predicted_label = (
            "реферируема DR"
            if decision["code"] == "provisional_positive"
            else "нереферируема DR"
        )
        decision["label"] = f"Моделен клас: {predicted_label}"
        decision["reason"] = (
            "Входът е pass по ръчно зададения quality gate и max(p, 1 - p) е "
            "над фиксирания праг. Показва се само предварителен моделен изход."
        )
    decision["policy_status"] = "experimental_not_clinically_validated"
    presented["decision"] = decision

    presented["technical_confidence"] = {
        "value": round(technical_confidence, 4),
        "definition": TECHNICAL_CONFIDENCE_DEFINITION,
        "interpretation": "Вероятност, дадена от калибрирания двоичен модел на избрания клас.",
        "limitation": TECHNICAL_CONFIDENCE_LIMITATION,
    }
    presented["limitations"] = [
        TECHNICAL_CONFIDENCE_LIMITATION,
        quality_gate_limitation(),
        "Изходът е изследователски и не е медицинска диагноза.",
    ]
    return presented


class RetinaHandler(BaseHTTPRequestHandler):
    server_version = "RetinaTrust/0.3"
    bundle: dict = {}

    def log_message(self, fmt: str, *args: object) -> None:
        logger.info("%s - %s", self.client_address[0], fmt % args)

    def _json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self._send_security_headers()
        self.end_headers()
        self.wfile.write(data)

    def _send_security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )

    def _serve_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self._send_security_headers()
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            robustness = compact_robustness()
            self._json(
                {
                    "status": "ready",
                    "model": self.bundle["metadata"],
                    "metrics": compact_metrics(self.bundle),
                    "robustness": robustness,
                    "supported_degradations": sorted(SUPPORTED_DEGRADATIONS),
                    "presentation": {
                        "technical_confidence_definition": TECHNICAL_CONFIDENCE_DEFINITION,
                        "technical_confidence_limitation": TECHNICAL_CONFIDENCE_LIMITATION,
                        "quality_gate_limitation": quality_gate_limitation(robustness),
                    },
                }
            )
            return
        if parsed.path == "/api/sample":
            if not DEMO_IMAGE.exists():
                self._json({"error": "Demo image is not installed."}, HTTPStatus.NOT_FOUND)
                return
            self._json({"name": DEMO_IMAGE.name, "image_data_url": image_to_data_url(load_rgb(DEMO_IMAGE))})
            return
        if parsed.path == "/health":
            self._json({"status": "ok"})
            return

        relative = "index.html" if parsed.path in ("", "/") else unquote(parsed.path.lstrip("/"))
        candidate = (STATIC / relative).resolve()
        try:
            candidate.relative_to(STATIC.resolve())
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        self._serve_file(candidate)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/analyze":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            validate_content_type(self.headers.get("Content-Type"))
            length = parse_content_length(self.headers.get("Content-Length"))
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValidationError("Заявката е прекъсната преди края на съдържанието.")
            payload = json.loads(body.decode("utf-8"))
            request, image = parse_and_validate_request(payload, length)
            original_size = [image.width, image.height]

            processed = apply_degradation(image, request.degradation, request.level)
            if request.normalize:
                processed = normalize_image(processed)

            result = present_case_result(predict_case(processed, self.bundle))
            result.update(
                {
                    "preview": image_to_data_url(processed),
                    "contrast_map": make_local_contrast_map(processed),
                    "input": {
                        "original_size": original_size,
                        "processed_size": [processed.width, processed.height],
                        "degradation": request.degradation,
                        "level": request.level,
                        "normalization": request.normalize,
                    },
                    "disclaimer": (
                        "Изследователски прототип. Изходът не е медицинска диагноза, "
                        "а quality gate и прагът за техническа увереност не са "
                        "клинично валидирани."
                    ),
                }
            )
            self._json(result)
        except ValidationError as exc:
            logger.warning("Rejected analysis request: %s", exc)
            self._json({"error": str(exc)}, exc.status_code)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
            logger.warning("Invalid JSON request: %s", exc)
            self._json({"error": "Невалидна JSON заявка."}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # keep server alive and avoid leaking a traceback to the browser
            logger.exception("Unexpected analysis error: %s", exc)
            self._json({"error": "Анализът не завърши. Проверете файла и опитайте отново."}, HTTPStatus.INTERNAL_SERVER_ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RetinaTrust local research PoC.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not ARTIFACT.exists():
        raise SystemExit("Model artifact is missing. Run: python scripts/train_baseline.py")
    RetinaHandler.bundle = load_bundle(ARTIFACT)
    server = ThreadingHTTPServer((args.host, args.port), RetinaHandler)
    url = f"http://{args.host}:{args.port}"
    logger.info("RetinaTrust is running at %s", url)
    logger.info("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping RetinaTrust.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
