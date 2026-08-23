#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from retina_poc.core import (  # noqa: E402
    CANONICAL_WORKING_SIZE,
    apply_degradation,
    canonicalize_image,
    extract_features,
    load_rgb,
    normalize_image,
)
from retina_poc.modeling import classification_metrics, load_bundle, score_quality  # noqa: E402


PERTURBATIONS = ("underexposure", "gamma", "color_shift", "blur", "jpeg")
BOOTSTRAP_SEED = 20260823


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the corrected IDRiD robustness matrix.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data/IDRiD/B. Disease Grading",
    )
    parser.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts/v0.3")
    parser.add_argument(
        "--legacy-artifacts",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
        help="v0.1 artifacts used only to quantify the old resampling confound.",
    )
    parser.add_argument("--with-normalization", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Development only; omit for the full official test split.",
    )
    return parser.parse_args()


def paired_comparison(
    y: np.ndarray,
    clean_probability: np.ndarray,
    condition_probability: np.ndarray,
    samples: int,
) -> dict[str, Any]:
    clean_metrics = classification_metrics(y, clean_probability)
    condition_metrics = classification_metrics(y, condition_probability)
    clean_prediction = (clean_probability >= 0.5).astype(int)
    condition_prediction = (condition_probability >= 0.5).astype(int)
    absolute_shift = np.abs(condition_probability - clean_probability)

    point = {
        "delta_auroc": condition_metrics["auroc"] - clean_metrics["auroc"],
        "delta_macro_f1": condition_metrics["macro_f1"] - clean_metrics["macro_f1"],
        "delta_balanced_accuracy": (
            condition_metrics["balanced_accuracy"] - clean_metrics["balanced_accuracy"]
        ),
        "prediction_flip_rate": float(np.mean(condition_prediction != clean_prediction)),
        "mean_absolute_probability_shift": float(np.mean(absolute_shift)),
        "mean_signed_probability_shift": float(np.mean(condition_probability - clean_probability)),
    }

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled: dict[str, list[float]] = {key: [] for key in point}
    for _ in range(samples):
        indices = rng.integers(0, len(y), len(y))
        sampled["prediction_flip_rate"].append(
            float(np.mean(condition_prediction[indices] != clean_prediction[indices]))
        )
        sampled["mean_absolute_probability_shift"].append(float(np.mean(absolute_shift[indices])))
        sampled["mean_signed_probability_shift"].append(
            float(np.mean(condition_probability[indices] - clean_probability[indices]))
        )
        if np.unique(y[indices]).size < 2:
            continue
        clean_sample = classification_metrics(y[indices], clean_probability[indices])
        condition_sample = classification_metrics(y[indices], condition_probability[indices])
        sampled["delta_auroc"].append(condition_sample["auroc"] - clean_sample["auroc"])
        sampled["delta_macro_f1"].append(
            condition_sample["macro_f1"] - clean_sample["macro_f1"]
        )
        sampled["delta_balanced_accuracy"].append(
            condition_sample["balanced_accuracy"] - clean_sample["balanced_accuracy"]
        )

    intervals = {
        key: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        for key, values in sampled.items()
        if values
    }
    return {**point, "paired_bootstrap_95_ci": intervals, "bootstrap_samples": int(samples)}


def condition_label(kind: str, level: int, normalise: bool) -> str:
    label = "clean" if kind == "none" else f"{kind}_level_{level}"
    return f"{label}_normalized" if normalise else label


def save_legacy_resampling_control(
    args: argparse.Namespace,
    labels: pd.DataFrame,
    y: np.ndarray,
    clean_features: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any] | None:
    legacy_model_path = args.legacy_artifacts / "retina_baseline.joblib"
    legacy_predictions_path = args.legacy_artifacts / "locked_test_predictions.csv"
    if not legacy_model_path.exists() or not legacy_predictions_path.exists():
        return None

    legacy_bundle = load_bundle(legacy_model_path)
    if list(feature_names) != list(legacy_bundle["feature_names"]):
        raise RuntimeError("Legacy feature schema mismatch")
    legacy_original = pd.read_csv(legacy_predictions_path).set_index("image_name")
    ordered_names = labels["Image name"].astype(str).tolist()
    legacy_original = legacy_original.loc[ordered_names]
    original_probability = legacy_original["probability_referable_dr"].to_numpy(dtype=float)
    resampling_probability = legacy_bundle["model"].predict_proba(clean_features)[:, 1]
    comparison = paired_comparison(
        y,
        original_probability,
        resampling_probability,
        samples=args.bootstrap_samples,
    )

    control_predictions = pd.DataFrame(
        {
            "image_name": ordered_names,
            "dr_grade": labels["Retinopathy grade"].astype(int).to_numpy(),
            "referable_dr": y,
            "legacy_clean_probability": original_probability,
            "resampling_only_probability": resampling_probability,
            "legacy_clean_prediction": (original_probability >= 0.5).astype(int),
            "resampling_only_prediction": (resampling_probability >= 0.5).astype(int),
            "prediction_flip": (
                (original_probability >= 0.5) != (resampling_probability >= 0.5)
            ).astype(int),
            "absolute_probability_shift": np.abs(resampling_probability - original_probability),
        }
    )
    control_predictions.to_csv(
        args.artifacts / "legacy_resampling_control_predictions.csv",
        index=False,
    )
    payload = {
        "purpose": (
            "Quantify the v0.1 confound: legacy direct clean path versus the same images passed "
            f"through the {CANONICAL_WORKING_SIZE}px resampling path without a named corruption."
        ),
        "n": int(len(y)),
        "legacy_clean_metrics": classification_metrics(y, original_probability),
        "resampling_only_metrics": classification_metrics(y, resampling_probability),
        "paired_comparison": comparison,
        "causal_interpretation": False,
    }
    (args.artifacts / "legacy_resampling_control.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    args = parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(args.artifacts / "retina_baseline.joblib")
    labels_path = args.data_root / "2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
    image_dir = args.data_root / "1. Original Images/b. Testing Set"
    labels = pd.read_csv(labels_path, usecols=["Image name", "Retinopathy grade"])
    if args.limit:
        labels = labels.iloc[: args.limit].copy()
    y = (labels["Retinopathy grade"].to_numpy() >= 2).astype(int)

    conditions: list[tuple[str, int, bool]] = [("none", 0, False)]
    conditions.extend((kind, level, False) for kind in PERTURBATIONS for level in (1, 2, 3))
    if args.with_normalization:
        conditions.extend((kind, level, True) for kind in PERTURBATIONS for level in (1, 2, 3))

    feature_rows: dict[tuple[str, int, bool], list[np.ndarray]] = defaultdict(list)
    quality_rows: dict[tuple[str, int, bool], list[dict[str, Any]]] = defaultdict(list)
    feature_names: list[str] | None = None

    for index, image_name in enumerate(labels["Image name"].astype(str), start=1):
        source = load_rgb(image_dir / f"{image_name}.jpg")
        canonical = canonicalize_image(source)
        for kind, level, normalise in conditions:
            transformed = canonical if kind == "none" else apply_degradation(canonical, kind, level)
            if normalise:
                transformed = normalize_image(transformed)
            vector, names, qmetrics = extract_features(transformed)
            if list(names) != list(bundle["feature_names"]):
                raise RuntimeError("Feature schema mismatch")
            if feature_names is None:
                feature_names = list(names)
            feature_rows[(kind, level, normalise)].append(vector)
            quality_rows[(kind, level, normalise)].append(
                score_quality(qmetrics, bundle["quality_reference"])
            )
        if index % 10 == 0 or index == len(labels):
            print(f"robustness images {index}/{len(labels)}", flush=True)

    matrices = {condition: np.vstack(rows) for condition, rows in feature_rows.items()}
    probabilities = {
        condition: bundle["model"].predict_proba(matrix)[:, 1]
        for condition, matrix in matrices.items()
    }
    clean_condition = ("none", 0, False)
    clean_probability = probabilities[clean_condition]
    clean_prediction = (clean_probability >= 0.5).astype(int)

    stored = pd.read_csv(args.artifacts / "locked_test_predictions.csv").set_index("image_name")
    stored = stored.loc[labels["Image name"].astype(str)]
    max_reproduction_error = float(
        np.max(np.abs(stored["probability_referable_dr"].to_numpy() - clean_probability))
    )
    if max_reproduction_error > 1e-10:
        raise RuntimeError(
            f"Clean prediction reproduction failed: max absolute error {max_reproduction_error}"
        )

    results: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for condition in conditions:
        kind, level, normalise = condition
        probability = probabilities[condition]
        prediction = (probability >= 0.5).astype(int)
        metrics = classification_metrics(y, probability)
        qualities = quality_rows[condition]
        gates = np.asarray([row["gate"] for row in qualities])
        comparison = (
            {
                "delta_auroc": 0.0,
                "delta_macro_f1": 0.0,
                "delta_balanced_accuracy": 0.0,
                "prediction_flip_rate": 0.0,
                "mean_absolute_probability_shift": 0.0,
                "mean_signed_probability_shift": 0.0,
                "paired_bootstrap_95_ci": {
                    key: [0.0, 0.0]
                    for key in (
                        "delta_auroc",
                        "delta_macro_f1",
                        "delta_balanced_accuracy",
                        "prediction_flip_rate",
                        "mean_absolute_probability_shift",
                        "mean_signed_probability_shift",
                    )
                },
                "bootstrap_samples": int(args.bootstrap_samples),
            }
            if condition == clean_condition
            else paired_comparison(y, clean_probability, probability, args.bootstrap_samples)
        )
        metrics.update(
            {
                "condition": kind,
                "condition_label": condition_label(kind, level, normalise),
                "level": level,
                "normalization": normalise,
                "mean_quality_score": float(np.mean([row["score"] for row in qualities])),
                "quality_fail_rate": float(np.mean(gates == "fail")),
                "quality_review_rate": float(np.mean(gates == "review")),
                "paired_vs_clean": comparison,
            }
        )
        results.append(metrics)

        for row_index, image_name in enumerate(labels["Image name"].astype(str)):
            quality = qualities[row_index]
            prediction_rows.append(
                {
                    "image_name": image_name,
                    "dr_grade": int(labels.iloc[row_index]["Retinopathy grade"]),
                    "referable_dr": int(y[row_index]),
                    "condition": kind,
                    "condition_label": condition_label(kind, level, normalise),
                    "level": level,
                    "normalization": normalise,
                    "probability_referable_dr": float(probability[row_index]),
                    "prediction": int(prediction[row_index]),
                    "confidence": float(max(probability[row_index], 1 - probability[row_index])),
                    "quality_score": float(quality["score"]),
                    "quality_gate": quality["gate"],
                    "prediction_flip_vs_clean": int(prediction[row_index] != clean_prediction[row_index]),
                    "absolute_probability_shift_vs_clean": float(
                        abs(probability[row_index] - clean_probability[row_index])
                    ),
                }
            )

    flat_rows: list[dict[str, Any]] = []
    for row in results:
        flat = {
            key: value
            for key, value in row.items()
            if key not in ("confusion_matrix", "paired_vs_clean")
        }
        comparison = row["paired_vs_clean"]
        for key, value in comparison.items():
            if key == "paired_bootstrap_95_ci":
                for metric_name, interval in value.items():
                    flat[f"{metric_name}_ci_low"] = interval[0]
                    flat[f"{metric_name}_ci_high"] = interval[1]
            elif key != "bootstrap_samples":
                flat[key] = value
        flat_rows.append(flat)
    frame = pd.DataFrame(flat_rows)
    frame.to_csv(args.artifacts / "robustness_results.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        args.artifacts / "robustness_predictions_long.csv",
        index=False,
    )

    legacy_control = save_legacy_resampling_control(
        args,
        labels,
        y,
        matrices[clean_condition],
        feature_names or [],
    )
    payload = {
        "protocol": {
            "version": "0.3.0",
            "dataset": "IDRiD official testing split",
            "n": int(len(labels)),
            "perturbations": list(PERTURBATIONS),
            "levels": [1, 2, 3],
            "canonical_max_dimension": CANONICAL_WORKING_SIZE,
            "shared_training_clean_degraded_path": True,
            "normalization_comparison": bool(args.with_normalization),
            "test_labels_used_for_fitting_or_threshold_tuning": False,
            "official_test_split_reused_after_v0.1_protocol_review": True,
            "paired_bootstrap_samples": int(args.bootstrap_samples),
            "clean_prediction_max_reproduction_error": max_reproduction_error,
            "legacy_resampling_control_available": legacy_control is not None,
        },
        "results": results,
    }
    (args.artifacts / "robustness_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        frame[
            [
                "condition",
                "level",
                "normalization",
                "auroc",
                "macro_f1",
                "mean_quality_score",
                "prediction_flip_rate",
                "delta_auroc",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
