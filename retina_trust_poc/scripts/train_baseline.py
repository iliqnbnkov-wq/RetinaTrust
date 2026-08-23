#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from retina_poc.core import CANONICAL_WORKING_SIZE, canonicalize_image, extract_features, load_rgb  # noqa: E402
from retina_poc.modeling import (  # noqa: E402
    build_quality_reference,
    classification_metrics,
    expected_calibration_error,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the locked IDRiD handcrafted-feature baseline.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data/IDRiD/B. Disease Grading",
    )
    parser.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts/v0.3")
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


def split_paths(data_root: Path, split: str) -> tuple[Path, Path]:
    if split == "train":
        return (
            data_root / "1. Original Images/a. Training Set",
            data_root / "2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv",
        )
    return (
        data_root / "1. Original Images/b. Testing Set",
        data_root / "2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv",
    )


def extract_split(data_root: Path, split: str) -> dict[str, object]:
    image_dir, label_path = split_paths(data_root, split)
    labels = pd.read_csv(label_path, usecols=["Image name", "Retinopathy grade"])
    rows: list[np.ndarray] = []
    quality_rows: list[dict[str, float]] = []
    names: list[str] | None = None
    targets: list[int] = []
    grades: list[int] = []
    image_names: list[str] = []

    total = len(labels)
    for index, row in labels.iterrows():
        image_name = str(row["Image name"])
        image_path = image_dir / f"{image_name}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        canonical = canonicalize_image(load_rgb(image_path))
        vector, current_names, qmetrics = extract_features(canonical)
        if names is None:
            names = current_names
        elif names != current_names:
            raise RuntimeError("Feature ordering changed between images")
        grade = int(row["Retinopathy grade"])
        rows.append(vector)
        quality_rows.append(qmetrics)
        targets.append(int(grade >= 2))
        grades.append(grade)
        image_names.append(image_name)
        if (index + 1) % 25 == 0 or index + 1 == total:
            print(f"[{split}] features {index + 1}/{total}", flush=True)

    return {
        "X": np.vstack(rows),
        "y": np.asarray(targets, dtype=np.int8),
        "grades": np.asarray(grades, dtype=np.int8),
        "image_names": np.asarray(image_names),
        "feature_names": np.asarray(names),
        "quality_rows": quality_rows,
    }


def save_cache(path: Path, data: dict[str, object]) -> None:
    quality_json = np.asarray([json.dumps(row, sort_keys=True) for row in data["quality_rows"]])
    np.savez_compressed(
        path,
        X=data["X"],
        y=data["y"],
        grades=data["grades"],
        image_names=data["image_names"],
        feature_names=data["feature_names"],
        quality_json=quality_json,
    )


def load_cache(path: Path) -> dict[str, object]:
    data = np.load(path, allow_pickle=False)
    return {
        "X": data["X"],
        "y": data["y"],
        "grades": data["grades"],
        "image_names": data["image_names"],
        "feature_names": data["feature_names"],
        "quality_rows": [json.loads(item) for item in data["quality_json"].tolist()],
    }


def bootstrap_intervals(y: np.ndarray, probability: np.ndarray, samples: int = 1000) -> dict[str, list[float]]:
    rng = np.random.default_rng(20260823)
    values: dict[str, list[float]] = {"auroc": [], "macro_f1": [], "balanced_accuracy": []}
    for _ in range(samples):
        indices = rng.integers(0, len(y), len(y))
        if np.unique(y[indices]).size < 2:
            continue
        metrics = classification_metrics(y[indices], probability[indices])
        for key in values:
            values[key].append(metrics[key])
    return {
        key: [float(np.percentile(item, 2.5)), float(np.percentile(item, 97.5))]
        for key, item in values.items()
    }


def risk_coverage(y: np.ndarray, probability: np.ndarray) -> list[dict[str, float]]:
    confidence = np.maximum(probability, 1 - probability)
    order = np.argsort(-confidence)
    prediction = (probability >= 0.5).astype(int)
    rows = []
    for coverage in np.linspace(0.1, 1.0, 10):
        count = max(1, int(round(len(y) * coverage)))
        keep = order[:count]
        risk = 1.0 - float(np.mean(prediction[keep] == y[keep]))
        rows.append({"coverage": round(float(count / len(y)), 4), "risk": round(risk, 4)})
    return rows


def choose_review_threshold(y: np.ndarray, probability: np.ndarray, target_risk: float = 0.15) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    confidence = np.maximum(probability, 1 - probability)
    candidates: list[dict[str, float]] = []
    for threshold in np.linspace(0.55, 0.95, 81):
        keep = confidence >= threshold
        if keep.mean() < 0.15:
            continue
        risk = 1.0 - float(np.mean(prediction[keep] == y[keep]))
        candidates.append(
            {"threshold": float(threshold), "coverage": float(keep.mean()), "risk": float(risk)}
        )
    feasible = [row for row in candidates if row["risk"] <= target_risk]
    if feasible:
        return max(feasible, key=lambda row: row["coverage"])
    return min(candidates, key=lambda row: row["risk"])


def main() -> None:
    args = parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, dict[str, object]] = {}
    for split in ("train", "test"):
        cache = args.artifacts / f"features_{split}_canonical{CANONICAL_WORKING_SIZE}.npz"
        if cache.exists() and not args.force_features:
            print(f"Loading {cache}")
            datasets[split] = load_cache(cache)
        else:
            datasets[split] = extract_split(args.data_root, split)
            save_cache(cache, datasets[split])

    train, test = datasets["train"], datasets["test"]
    X_train, y_train = train["X"], train["y"]
    X_test, y_test = test["X"], test["y"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260823)
    pipeline = Pipeline(
        [
            ("scale", RobustScaler(quantile_range=(10, 90))),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=20260823,
                ),
            ),
        ]
    )
    search = GridSearchCV(
        pipeline,
        {"classifier__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    print(f"Best C={search.best_params_['classifier__C']} CV AUROC={search.best_score_:.4f}")

    oof_calibrated = CalibratedClassifierCV(estimator=search.best_estimator_, method="sigmoid", cv=3)
    oof_probability = cross_val_predict(
        oof_calibrated, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]
    review_policy = choose_review_threshold(y_train, oof_probability)
    # The interactive PoC uses a deliberately conservative design floor. It is
    # not a clinically validated cut-off and is not optimised on the official test split.
    deployed_review_threshold = max(float(review_policy["threshold"]), 0.85)
    print(
        "OOF review policy "
        f"confidence>={review_policy['threshold']:.3f}, "
        f"coverage={review_policy['coverage']:.3f}, risk={review_policy['risk']:.3f}"
    )

    calibrated = CalibratedClassifierCV(estimator=search.best_estimator_, method="sigmoid", cv=cv)
    calibrated.fit(X_train, y_train)
    probability = calibrated.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, probability)
    metrics["bootstrap_95_ci"] = bootstrap_intervals(y_test, probability)
    metrics["risk_coverage"] = risk_coverage(y_test, probability)
    metrics["model_selection"] = {
        "best_C": float(search.best_params_["classifier__C"]),
        "internal_cv_auroc": float(search.best_score_),
        "oof_ece": expected_calibration_error(y_train, oof_probability),
        "review_policy_from_oof": review_policy,
        "deployed_confidence_floor": deployed_review_threshold,
    }
    metrics["data"] = {
        "dataset": "IDRiD B. Disease Grading",
        "train_images": int(len(y_train)),
        "locked_test_images": int(len(y_test)),
        "target": "Referable DR (grade >= 2)",
        "train_positive": int(y_train.sum()),
        "test_positive": int(y_test.sum()),
        "preprocessing": f"Shared max-dimension {CANONICAL_WORKING_SIZE}px LANCZOS path before feature extraction",
        "official_test_split_reused_after_protocol_correction": True,
        "test_labels_used_for_model_fitting_or_threshold_selection": False,
    }

    reference = build_quality_reference(train["quality_rows"])
    metadata = {
        "name": "IDRiD handcrafted-feature calibrated baseline",
        "version": "0.3.0",
        "target": "Referable DR (grade >= 2)",
        "training_images": int(len(y_train)),
        "locked_test_images": int(len(y_test)),
        "feature_count": int(X_train.shape[1]),
        "clinical_use": False,
        "preprocessing": {
            "canonical_max_dimension": CANONICAL_WORKING_SIZE,
            "resampling": "Pillow LANCZOS",
            "shared_across_training_clean_and_degraded_paths": True,
        },
        "limitations": (
            "Single-centre public dataset; official test split was reused after a protocol correction; "
            "research demonstration only."
        ),
    }
    bundle = {
        "model": calibrated,
        "feature_names": train["feature_names"].tolist(),
        "quality_reference": reference,
        "classification_threshold": 0.5,
        "review_confidence_threshold": deployed_review_threshold,
        "metadata": metadata,
        "locked_test_metrics": metrics,
    }
    joblib.dump(bundle, args.artifacts / "retina_baseline.joblib", compress=3)
    (args.artifacts / "locked_test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    predictions = pd.DataFrame(
        {
            "image_name": test["image_names"],
            "dr_grade": test["grades"],
            "referable_dr": y_test,
            "probability_referable_dr": probability,
            "confidence": np.maximum(probability, 1 - probability),
        }
    )
    predictions.to_csv(args.artifacts / "locked_test_predictions.csv", index=False)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
