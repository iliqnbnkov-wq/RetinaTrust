from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .core import canonicalize_image, extract_features, load_rgb
from .modeling import expected_calibration_error, score_quality


DEEPDRID_REQUIRED_COLUMNS = (
    "patient_id",
    "image_id",
    "image_path",
    "Overall quality",
    "left_eye_DR_Level",
    "right_eye_DR_Level",
    "patient_DR_Level",
    "Clarity",
    "Field definition",
    "Artifact",
)
DEEPDRID_PARTITIONS = {
    "training": "regular-fundus-training",
    "validation": "regular-fundus-validation",
}
VALID_DR_GRADES = frozenset(range(6))
UNGRADABLE_DR_GRADE = 5
DEFAULT_BOOTSTRAP_SEED = 20260823
_EYE_PATTERN = re.compile(r"_(?P<eye>[lr])[12]$", re.IGNORECASE)


@dataclass(frozen=True)
class DeepDRiDRecord:
    partition: str
    patient_id: str
    image_id: str
    image_path: Path
    eye: str
    dr_grade: int
    patient_dr_grade: int
    quality_good: int
    clarity: int
    field_definition: int
    artifact: int

    @property
    def referable_dr(self) -> int | None:
        if self.dr_grade == UNGRADABLE_DR_GRADE:
            return None
        return int(self.dr_grade >= 2)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_identifier(value: Any, field: str, row_number: int) -> str:
    if pd.isna(value):
        raise ValueError(f"Row {row_number}: {field} is missing.")
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text:
        raise ValueError(f"Row {row_number}: {field} is empty.")
    return text


def _parse_integer(
    value: Any,
    field: str,
    row_number: int,
    allowed: set[int] | frozenset[int] | None = None,
) -> int:
    if pd.isna(value):
        raise ValueError(f"Row {row_number}: {field} is missing.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Row {row_number}: {field} is not numeric.") from exc
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(f"Row {row_number}: {field} must be an integer.")
    result = int(number)
    if allowed is not None and result not in allowed:
        allowed_text = ", ".join(str(item) for item in sorted(allowed))
        raise ValueError(f"Row {row_number}: {field} must be one of {allowed_text}.")
    return result


def _optional_integer(value: Any, field: str, row_number: int) -> int | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    return _parse_integer(value, field, row_number, VALID_DR_GRADES)


def _resolve_image_path(
    label_csv: Path,
    patient_id: str,
    image_id: str,
    raw_path: Any,
) -> Path:
    raw_text = "" if pd.isna(raw_path) else str(raw_path).strip()
    filename = PureWindowsPath(raw_text).name if raw_text else f"{image_id}.jpg"
    if filename in ("", ".", ".."):
        filename = f"{image_id}.jpg"
    if Path(filename).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(f"Unsupported DeepDRiD image suffix for {image_id}: {filename}")

    image_root = label_csv.parent / "Images"
    candidates = (
        image_root / patient_id / filename,
        image_root / filename,
        label_csv.parent / patient_id / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def read_deepdrid_csv(
    label_csv: str | Path,
    *,
    require_images: bool = True,
) -> list[DeepDRiDRecord]:
    """Parse one official DeepDRiD regular-fundus CSV with strict schema checks."""
    csv_path = Path(label_csv).resolve()
    frame = pd.read_csv(csv_path)
    missing = [column for column in DEEPDRID_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path.name}: missing columns: {', '.join(missing)}")

    partition = csv_path.parent.name
    records: list[DeepDRiDRecord] = []
    seen_images: set[str] = set()
    for index, row in frame.iterrows():
        row_number = int(index) + 2
        patient_id = _normalise_identifier(row["patient_id"], "patient_id", row_number)
        image_id = _normalise_identifier(row["image_id"], "image_id", row_number)
        if image_id in seen_images:
            raise ValueError(f"Row {row_number}: duplicate image_id {image_id}.")
        seen_images.add(image_id)

        match = _EYE_PATTERN.search(image_id)
        if match is None:
            raise ValueError(f"Row {row_number}: cannot infer eye from image_id {image_id}.")
        eye = match.group("eye").lower()
        left_grade = _optional_integer(row["left_eye_DR_Level"], "left_eye_DR_Level", row_number)
        right_grade = _optional_integer(row["right_eye_DR_Level"], "right_eye_DR_Level", row_number)
        if eye == "l" and (left_grade is None or right_grade is not None):
            raise ValueError(f"Row {row_number}: left image has inconsistent eye-grade columns.")
        if eye == "r" and (right_grade is None or left_grade is not None):
            raise ValueError(f"Row {row_number}: right image has inconsistent eye-grade columns.")
        dr_grade = left_grade if eye == "l" else right_grade
        if dr_grade is None:  # guarded above; keeps the type explicit
            raise ValueError(f"Row {row_number}: eye DR grade is missing.")

        quality_good = _parse_integer(
            row["Overall quality"], "Overall quality", row_number, {0, 1}
        )
        patient_grade = _parse_integer(
            row["patient_DR_Level"], "patient_DR_Level", row_number, VALID_DR_GRADES
        )
        clarity = _parse_integer(row["Clarity"], "Clarity", row_number)
        field_definition = _parse_integer(
            row["Field definition"], "Field definition", row_number
        )
        artifact = _parse_integer(row["Artifact"], "Artifact", row_number)
        for name, value in (
            ("Clarity", clarity),
            ("Field definition", field_definition),
            ("Artifact", artifact),
        ):
            if value < 0 or value > 10:
                raise ValueError(f"Row {row_number}: {name} must be between 0 and 10.")

        image_path = _resolve_image_path(csv_path, patient_id, image_id, row["image_path"])
        if require_images and not image_path.is_file():
            raise FileNotFoundError(
                f"Row {row_number}: image not found at expected path {image_path}"
            )
        records.append(
            DeepDRiDRecord(
                partition=partition,
                patient_id=patient_id,
                image_id=image_id,
                image_path=image_path,
                eye=eye,
                dr_grade=dr_grade,
                patient_dr_grade=patient_grade,
                quality_good=quality_good,
                clarity=clarity,
                field_definition=field_definition,
                artifact=artifact,
            )
        )
    return records


def discover_deepdrid_label_csvs(
    dataset_root: str | Path,
    partitions: Sequence[str] = ("training", "validation"),
) -> list[Path]:
    root = Path(dataset_root).resolve()
    regular_root = root / "regular_fundus_images" if (root / "regular_fundus_images").is_dir() else root
    unknown = sorted(set(partitions) - set(DEEPDRID_PARTITIONS))
    if unknown:
        raise ValueError(f"Unsupported DeepDRiD partitions: {', '.join(unknown)}")

    paths: list[Path] = []
    for partition in partitions:
        directory = DEEPDRID_PARTITIONS[partition]
        candidate = regular_root / directory / f"{directory}.csv"
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Missing official DeepDRiD {partition} labels: {candidate}"
            )
        paths.append(candidate)
    return paths


def load_deepdrid_records(
    dataset_root: str | Path,
    partitions: Sequence[str] = ("training", "validation"),
    *,
    require_images: bool = True,
) -> tuple[list[DeepDRiDRecord], list[Path]]:
    csv_paths = discover_deepdrid_label_csvs(dataset_root, partitions)
    records: list[DeepDRiDRecord] = []
    seen: set[str] = set()
    for csv_path in csv_paths:
        for record in read_deepdrid_csv(csv_path, require_images=require_images):
            if record.image_id in seen:
                raise ValueError(f"Duplicate image_id across partitions: {record.image_id}")
            seen.add(record.image_id)
            records.append(record)
    return records, csv_paths


def predict_external_record(record: DeepDRiDRecord, bundle: dict[str, Any]) -> dict[str, Any]:
    """Run the frozen single-image model without rounding probabilities."""
    image = canonicalize_image(load_rgb(record.image_path))
    vector, names, quality_metrics = extract_features(image)
    if list(names) != list(bundle["feature_names"]):
        raise RuntimeError("Feature schema does not match the frozen model.")
    probability = float(bundle["model"].predict_proba(vector.reshape(1, -1))[0, 1])
    quality = score_quality(quality_metrics, bundle["quality_reference"])
    confidence = float(max(probability, 1.0 - probability))
    classification_threshold = float(bundle.get("classification_threshold", 0.5))
    confidence_threshold = float(bundle.get("review_confidence_threshold", 0.85))
    autonomous = quality["gate"] == "pass" and confidence >= confidence_threshold
    return {
        "partition": record.partition,
        "patient_id": record.patient_id,
        "image_id": record.image_id,
        "eye": record.eye,
        "dr_grade": record.dr_grade,
        "patient_dr_grade": record.patient_dr_grade,
        "referable_dr": record.referable_dr,
        "quality_good": record.quality_good,
        "clarity": record.clarity,
        "field_definition": record.field_definition,
        "artifact": record.artifact,
        "probability_referable_dr": probability,
        "confidence": confidence,
        "predicted_referable_dr": int(probability >= classification_threshold),
        "quality_score": float(quality["score"]),
        "quality_gate": quality["gate"],
        "autonomous_eligible": bool(autonomous),
    }


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def binary_probability_metrics(
    y_true: Iterable[int],
    probability: Iterable[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    y = np.asarray(list(y_true), dtype=int)
    score = np.asarray(list(probability), dtype=float)
    if y.ndim != 1 or score.ndim != 1 or len(y) != len(score) or len(y) == 0:
        raise ValueError("y_true and probability must be non-empty, equal-length vectors.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1.")
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError("probabilities must be finite and between 0 and 1.")

    prediction = (score >= threshold).astype(int)
    tn = int(np.sum((y == 0) & (prediction == 0)))
    fp = int(np.sum((y == 0) & (prediction == 1)))
    fn = int(np.sum((y == 1) & (prediction == 0)))
    tp = int(np.sum((y == 1) & (prediction == 1)))
    sensitivity = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    precision = _safe_ratio(tp, tp + fp)
    npv = _safe_ratio(tn, tn + fn)
    f1_positive = _safe_ratio(2 * tp, 2 * tp + fp + fn) or 0.0
    f1_negative = _safe_ratio(2 * tn, 2 * tn + fp + fn) or 0.0
    eps = 1e-12
    clipped = np.clip(score, eps, 1 - eps)
    nll = -float(np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))
    auroc = float(roc_auc_score(y, score)) if np.unique(y).size == 2 else None
    balanced = (
        float((sensitivity + specificity) / 2)
        if sensitivity is not None and specificity is not None
        else None
    )
    return {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
        "threshold": float(threshold),
        "auroc": auroc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision_ppv": precision,
        "negative_predictive_value": npv,
        "f1": float(f1_positive),
        "macro_f1": float((f1_positive + f1_negative) / 2),
        "balanced_accuracy": balanced,
        "accuracy": float(np.mean(prediction == y)),
        "brier": float(np.mean((score - y) ** 2)),
        "nll": nll,
        "ece": expected_calibration_error(y, score),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def binary_decision_metrics(y_true: Iterable[int], prediction: Iterable[int]) -> dict[str, Any]:
    y = np.asarray(list(y_true), dtype=int)
    predicted = np.asarray(list(prediction), dtype=int)
    if y.ndim != 1 or predicted.ndim != 1 or len(y) != len(predicted) or len(y) == 0:
        raise ValueError("y_true and prediction must be non-empty, equal-length vectors.")
    if not np.isin(y, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
        raise ValueError("binary decision metrics accept only 0 and 1.")
    tn = int(np.sum((y == 0) & (predicted == 0)))
    fp = int(np.sum((y == 0) & (predicted == 1)))
    fn = int(np.sum((y == 1) & (predicted == 0)))
    tp = int(np.sum((y == 1) & (predicted == 1)))
    sensitivity = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    precision = _safe_ratio(tp, tp + fp)
    npv = _safe_ratio(tn, tn + fn)
    f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn) or 0.0
    return {
        "n": int(len(y)),
        "true_positive_event": "officially ungradable image (Overall quality = 0)",
        "sensitivity_ungradable": sensitivity,
        "specificity_gradable": specificity,
        "precision_ppv_ungradable": precision,
        "negative_predictive_value_gradable": npv,
        "f1_ungradable": float(f1),
        "balanced_accuracy": (
            float((sensitivity + specificity) / 2)
            if sensitivity is not None and specificity is not None
            else None
        ),
        "under_rejection_rate": _safe_ratio(fn, tp + fn),
        "over_rejection_rate": _safe_ratio(fp, tn + fp),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def quality_gate_metrics(quality_good: Iterable[int], gates: Iterable[str]) -> dict[str, Any]:
    official_good = np.asarray(list(quality_good), dtype=int)
    gate = np.asarray(list(gates), dtype=str)
    if len(official_good) == 0 or len(official_good) != len(gate):
        raise ValueError("quality_good and gates must be non-empty and equal in length.")
    if not np.isin(official_good, [0, 1]).all():
        raise ValueError("quality_good must contain only 0 and 1.")
    if not np.isin(gate, ["pass", "review", "fail"]).all():
        raise ValueError("gates must contain pass, review or fail.")
    official_ungradable = 1 - official_good
    intervention = (gate != "pass").astype(int)
    recapture = (gate == "fail").astype(int)
    return {
        "n": int(len(gate)),
        "official_gradable": int(official_good.sum()),
        "official_ungradable": int(official_ungradable.sum()),
        "gate_distribution": {
            name: int(np.sum(gate == name)) for name in ("pass", "review", "fail")
        },
        "pass_coverage": float(np.mean(gate == "pass")),
        "intervention_review_or_fail": binary_decision_metrics(
            official_ungradable, intervention
        ),
        "recapture_fail_only": binary_decision_metrics(official_ungradable, recapture),
    }


def cluster_bootstrap_intervals(
    y_true: Iterable[int],
    probability: Iterable[float],
    clusters: Iterable[str],
    *,
    threshold: float = 0.5,
    samples: int = 2000,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    y = np.asarray(list(y_true), dtype=int)
    score = np.asarray(list(probability), dtype=float)
    cluster = np.asarray([str(item) for item in clusters], dtype=str)
    if len(y) != len(score) or len(y) != len(cluster) or len(y) == 0:
        raise ValueError("y_true, probability and clusters must have equal non-zero length.")
    if samples < 1:
        raise ValueError("samples must be positive.")
    unique_clusters = np.unique(cluster)
    if len(unique_clusters) < 2:
        raise ValueError("At least two patient clusters are required.")

    tracked = (
        "auroc",
        "sensitivity",
        "specificity",
        "macro_f1",
        "balanced_accuracy",
        "brier",
        "nll",
        "ece",
    )
    values: dict[str, list[float]] = {key: [] for key in tracked}
    cluster_indices = {item: np.flatnonzero(cluster == item) for item in unique_clusters}
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        selected = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        indices = np.concatenate([cluster_indices[item] for item in selected])
        metrics = binary_probability_metrics(y[indices], score[indices], threshold)
        for key in tracked:
            value = metrics[key]
            if value is not None and np.isfinite(value):
                values[key].append(float(value))

    intervals: dict[str, Any] = {}
    for key, observed in values.items():
        intervals[key] = {
            "95_ci": (
                [float(np.percentile(observed, 2.5)), float(np.percentile(observed, 97.5))]
                if observed
                else None
            ),
            "valid_replicates": int(len(observed)),
        }
    return {
        "method": "patient-clustered nonparametric bootstrap",
        "cluster_count": int(len(unique_clusters)),
        "samples_requested": int(samples),
        "seed": int(seed),
        "intervals": intervals,
    }


def aggregate_eye_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "patient_id",
        "eye",
        "dr_grade",
        "probability_referable_dr",
        "quality_gate",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {', '.join(sorted(missing))}")
    rows: list[dict[str, Any]] = []
    for (patient_id, eye), group in frame.groupby(["patient_id", "eye"], sort=True):
        grades = sorted(set(int(value) for value in group["dr_grade"]))
        if len(grades) != 1:
            raise ValueError(f"Inconsistent grades for patient {patient_id}, eye {eye}.")
        grade = grades[0]
        rows.append(
            {
                "patient_id": str(patient_id),
                "eye": str(eye),
                "dr_grade": grade,
                "referable_dr": None if grade == UNGRADABLE_DR_GRADE else int(grade >= 2),
                "probability_referable_dr": float(group["probability_referable_dr"].max()),
                "view_count": int(len(group)),
                "has_quality_pass_view": bool((group["quality_gate"] == "pass").any()),
            }
        )
    return pd.DataFrame(rows)


def aggregate_patient_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"patient_id", "patient_dr_grade", "probability_referable_dr"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {', '.join(sorted(missing))}")
    rows: list[dict[str, Any]] = []
    for patient_id, group in frame.groupby("patient_id", sort=True):
        grades = sorted(set(int(value) for value in group["patient_dr_grade"]))
        if len(grades) != 1:
            raise ValueError(f"Inconsistent patient grade for patient {patient_id}.")
        grade = grades[0]
        rows.append(
            {
                "patient_id": str(patient_id),
                "patient_dr_grade": grade,
                "referable_dr": None if grade == UNGRADABLE_DR_GRADE else int(grade >= 2),
                "probability_referable_dr": float(group["probability_referable_dr"].max()),
                "image_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def grade_stratified_errors(frame: pd.DataFrame, threshold: float = 0.5) -> list[dict[str, Any]]:
    required = {"dr_grade", "probability_referable_dr", "quality_gate"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {', '.join(sorted(missing))}")
    rows: list[dict[str, Any]] = []
    for grade in range(5):
        group = frame[frame["dr_grade"] == grade]
        if group.empty:
            rows.append({"dr_grade": grade, "n": 0})
            continue
        predicted = (group["probability_referable_dr"].to_numpy(float) >= threshold).astype(int)
        expected = int(grade >= 2)
        errors = int(np.sum(predicted != expected))
        rows.append(
            {
                "dr_grade": grade,
                "n": int(len(group)),
                "referable_target": expected,
                "mean_probability_referable_dr": float(
                    group["probability_referable_dr"].mean()
                ),
                "predicted_positive_rate": float(np.mean(predicted)),
                "error_count": errors,
                "error_rate": float(errors / len(group)),
                "quality_pass_coverage": float(np.mean(group["quality_gate"] == "pass")),
            }
        )
    return rows


def verify_protocol_lock(
    lock_path: str | Path,
    project_root: str | Path,
    bundle: dict[str, Any],
    model_path: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    path = Path(lock_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Protocol lock must be inside the project root.") from exc
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported external-validation protocol-lock schema.")
    if payload.get("status") != "frozen_before_external_results":
        raise ValueError("Protocol lock is not marked as frozen before external results.")

    locked_model = payload.get("model_path")
    if not isinstance(locked_model, str):
        raise ValueError("Protocol lock does not identify the frozen model path.")
    expected_model_path = (root / locked_model).resolve()
    if Path(model_path).resolve() != expected_model_path:
        raise ValueError("Requested model is not the model frozen by the protocol lock.")

    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Protocol lock contains no file hashes.")
    for entry in entries:
        relative = entry.get("path") if isinstance(entry, dict) else None
        expected = entry.get("sha256") if isinstance(entry, dict) else None
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("Invalid protocol-lock file entry.")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Protocol-lock path escapes project root: {relative}") from exc
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise ValueError(f"Protocol-lock hash mismatch: {relative}")

    thresholds = payload.get("thresholds", {})
    expected_classification = float(thresholds.get("classification", -1))
    expected_confidence = float(thresholds.get("confidence_floor", -1))
    if expected_classification != float(bundle.get("classification_threshold", 0.5)):
        raise ValueError("Classification threshold differs from the frozen protocol.")
    if expected_confidence != float(bundle.get("review_confidence_threshold", 0.85)):
        raise ValueError("Confidence floor differs from the frozen protocol.")
    return payload
