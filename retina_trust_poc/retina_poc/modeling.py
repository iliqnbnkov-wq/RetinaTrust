from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)

from .core import extract_features, image_to_data_url, prepare_image


QUALITY_LABELS = {
    "brightness": "Яркост",
    "contrast": "Контраст",
    "sharpness": "Острота",
    "field_coverage": "Покритие на полето",
    "illumination_nonuniformity": "Равномерност",
    "color_cast": "Цветови баланс",
    "clipping": "Запазени тонове",
}


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        include = (probability >= low) & (probability < high if high < 1 else probability <= high)
        if not include.any():
            continue
        accuracy = y_true[include].mean()
        confidence = probability[include].mean()
        result += include.mean() * abs(accuracy - confidence)
    return float(result)


def classification_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    eps = 1e-12
    return {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y_true, probability)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "brier": float(brier_score_loss(y_true, probability)),
        "nll": float(log_loss(y_true, np.clip(probability, eps, 1 - eps), labels=[0, 1])),
        "ece": expected_calibration_error(y_true, probability),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def build_quality_reference(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted(rows[0])
    reference: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=float)
        reference[key] = {
            f"q{p:02d}": float(np.percentile(values, p))
            for p in (1, 5, 25, 50, 75, 95, 99)
        }
    return reference


def _two_sided_score(value: float, stats: dict[str, float]) -> float:
    median = stats["q50"]
    scale = median - stats["q05"] if value < median else stats["q95"] - median
    scale = max(scale, 1e-6)
    z = abs(value - median) / scale
    return float(np.clip(100.0 * np.exp(-0.36 * z * z), 0, 100))


def _lower_is_bad_score(value: float, stats: dict[str, float]) -> float:
    median = stats["q50"]
    if value >= median:
        return 100.0
    z = (median - value) / max(median - stats["q05"], 1e-6)
    return float(np.clip(100.0 * np.exp(-0.42 * z * z), 0, 100))


def _higher_is_bad_score(value: float, stats: dict[str, float]) -> float:
    median = stats["q50"]
    if value <= median:
        return 100.0
    z = (value - median) / max(stats["q95"] - median, 1e-6)
    return float(np.clip(100.0 * np.exp(-0.42 * z * z), 0, 100))


def score_quality(metrics: dict[str, float], reference: dict[str, dict[str, float]]) -> dict[str, Any]:
    dark = _higher_is_bad_score(metrics["dark_clipping"], reference["dark_clipping"])
    bright = _higher_is_bad_score(metrics["bright_clipping"], reference["bright_clipping"])
    component_scores = {
        "brightness": _two_sided_score(metrics["brightness"], reference["brightness"]),
        "contrast": _lower_is_bad_score(metrics["contrast"], reference["contrast"]),
        "sharpness": _lower_is_bad_score(metrics["sharpness"], reference["sharpness"]),
        # Field coverage is scored against an absolute geometric range. Its
        # training-set distribution is too narrow and would incorrectly reject
        # the same image after a harmless resize or re-encoding.
        "field_coverage": float(np.clip((metrics["field_coverage"] - 0.45) / 0.30 * 100, 0, 100)),
        "illumination_nonuniformity": _higher_is_bad_score(
            metrics["illumination_nonuniformity"], reference["illumination_nonuniformity"]
        ),
        "color_cast": min(
            _two_sided_score(metrics["color_cast"], reference["color_cast"]),
            _two_sided_score(metrics["saturation"], reference["saturation"]),
        ),
        "clipping": min(dark, bright),
    }
    weights = {
        "brightness": 0.18,
        "contrast": 0.15,
        "sharpness": 0.22,
        "field_coverage": 0.15,
        "illumination_nonuniformity": 0.10,
        "color_cast": 0.10,
        "clipping": 0.10,
    }
    overall = sum(component_scores[key] * weights[key] for key in weights)
    weakest = min(component_scores, key=component_scores.get)

    if overall < 45 or min(component_scores["brightness"], component_scores["sharpness"], component_scores["field_coverage"]) < 30:
        gate = "fail"
    elif overall < 70 or component_scores[weakest] < 65:
        gate = "review"
    else:
        gate = "pass"

    issues = [
        {"key": key, "label": QUALITY_LABELS[key], "score": round(value, 1)}
        for key, value in sorted(component_scores.items(), key=lambda item: item[1])
        if value < 70
    ]
    return {
        "score": round(float(overall), 1),
        "gate": gate,
        "components": {key: round(float(value), 1) for key, value in component_scores.items()},
        "issues": issues,
    }


def load_bundle(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)


def make_local_contrast_map(image: Image.Image) -> str:
    prepared = prepare_image(image)
    lum = 0.2126 * prepared.array[..., 0] + 0.7152 * prepared.array[..., 1] + 0.0722 * prepared.array[..., 2]
    detail = np.abs(lum - ndimage.gaussian_filter(lum, sigma=4.0))
    scale = max(float(np.percentile(detail[prepared.mask], 98)), 1e-6)
    heat = np.clip(detail / scale, 0, 1)
    red = np.clip(1.7 * heat, 0, 1)
    green = np.clip(1.7 * (1 - np.abs(heat - .5) * 2), 0, 1)
    blue = np.clip(1.4 * (1 - heat), 0, 1)
    colour = np.stack([red, green, blue], axis=-1)
    alpha = (0.15 + 0.55 * heat)[..., None]
    overlay = prepared.array * (1 - alpha) + colour * alpha
    overlay[~prepared.mask] = prepared.array[~prepared.mask]
    output = Image.fromarray(np.uint8(np.clip(overlay, 0, 1) * 255), "RGB")
    return image_to_data_url(output)


def predict_case(image: Image.Image, bundle: dict[str, Any]) -> dict[str, Any]:
    vector, names, qmetrics = extract_features(image)
    expected_names = bundle["feature_names"]
    if list(names) != list(expected_names):
        raise RuntimeError("Feature schema does not match the trained model")
    probability = float(bundle["model"].predict_proba(vector.reshape(1, -1))[0, 1])
    confidence = float(max(probability, 1 - probability))
    uncertainty = float(1 - confidence)
    quality = score_quality(qmetrics, bundle["quality_reference"])
    threshold = float(bundle.get("classification_threshold", 0.5))
    confidence_threshold = float(bundle.get("review_confidence_threshold", 0.70))

    if quality["gate"] == "fail":
        decision_code = "recapture"
        decision = "Повторно заснемане"
        reason = "Качеството е недостатъчно за надеждна автоматична оценка."
    elif quality["gate"] == "review" or confidence < confidence_threshold:
        decision_code = "review"
        decision = "Човешки преглед"
        reason = "Качеството или увереността не покриват предварително зададения праг."
    else:
        decision_code = "provisional_positive" if probability >= threshold else "provisional_negative"
        decision = "Предварително: реферируема DR" if probability >= threshold else "Предварително: нереферируема DR"
        reason = "Технически приемливо изображение и достатъчна увереност на демонстрационния модел."

    raw_metrics = {
        key: round(float(value), 5)
        for key, value in qmetrics.items()
    }
    return {
        "probability_referable_dr": round(probability, 4),
        "confidence": round(confidence, 4),
        "uncertainty": round(uncertainty, 4),
        "predicted_class": int(probability >= threshold),
        "quality": quality,
        "raw_quality_metrics": raw_metrics,
        "decision": {"code": decision_code, "label": decision, "reason": reason},
        "thresholds": {"classification": threshold, "confidence": confidence_threshold},
        "model": bundle["metadata"],
    }


def save_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
