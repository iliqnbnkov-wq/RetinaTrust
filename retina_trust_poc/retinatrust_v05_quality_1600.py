#!/usr/bin/env python3
"""RetinaTrust v0.5: auditable post-hoc completion of DeepDRiD quality analysis.

Preserves the frozen diagnostic evaluation (1589 records), carries over its
stored quality decisions, and computes quality ONLY for the 11 auxiliary views
that have no image-specific DR labels.  Never trains or changes the model.

Example (PowerShell):
 python -u retinatrust_v05_quality_1600.py --project-root "C:\\Users\\User\\RetinaTrust-final-check\\retina_trust_poc" --dataset-root "C:\\Users\\User\\RetinaTrust-data\\DeepDRiD\\deepdrdoc-DeepDRiD-56d8af7"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EXCLUDED = frozenset({
    "56_r3", "56_l3", "77_r4", "77_r3", "77_l3", "77_l4",
    "164_r3", "164_r4", "164_l3", "167_r3", "167_l3",
})
PARTITIONS = ("regular-fundus-training", "regular-fundus-validation")
REQUIRED_CSV = {"patient_id", "image_id", "image_path", "Overall quality",
                "left_eye_DR_Level", "right_eye_DR_Level", "patient_DR_Level",
                "Clarity", "Field definition", "Artifact"}
REQUIRED_PREDICTIONS = {"partition", "patient_id", "image_id", "quality_good",
                        "quality_score", "quality_gate"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_lock(project_root: Path) -> tuple[dict, Path]:
    lock_file = project_root / "artifacts/v0.4/external_protocol_lock.json"
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    ensure(lock.get("schema_version") == 1 and lock.get("status") == "frozen_before_external_results",
           "Unexpected protocol-lock status or schema")
    for entry in lock["files"]:
        candidate = (project_root / entry["path"]).resolve()
        ensure(candidate.is_relative_to(project_root), f"Lock path escapes project root: {entry['path']}")
        ensure(candidate.is_file(), f"Missing locked file: {candidate}")
        ensure(sha256(candidate).lower() == entry["sha256"].lower(),
               f"Frozen-file SHA256 mismatch: {entry['path']}")
    model_file = (project_root / lock["model_path"]).resolve()
    ensure(model_file.is_relative_to(project_root), "Model path escapes project root")
    ensure(model_file.is_file(), f"Frozen model missing: {model_file}")
    return lock, lock_file


def normalise_identifier(value: object) -> str:
    ensure(pd.notna(value), "Missing patient/image identifier")
    return str(value).strip()


def resolve_image(csv_path: Path, patient_id: str, image_id: str, raw_path: object) -> Path:
    raw_text = "" if pd.isna(raw_path) else str(raw_path).strip()
    filename = PureWindowsPath(raw_text).name if raw_text else f"{image_id}.jpg"
    ensure(Path(filename).suffix.lower() in (".jpg", ".jpeg", ".png"),
           f"Unsupported image filename: {image_id}: {filename}")
    candidates = [csv_path.parent / "Images" / patient_id / filename,
                  csv_path.parent / "Images" / filename,
                  csv_path.parent / patient_id / filename]
    matches = [p for p in candidates if p.is_file()]
    ensure(len(matches) >= 1, f"Image not found for {image_id}; tried {candidates}")
    return matches[0]


def value_or_none(value: object) -> int | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    number = float(value)
    ensure(number.is_integer(), f"Unexpected noninteger DR grade {value}")
    return int(number)


def quality_stats(rows: pd.DataFrame, prediction_kind: str) -> dict:
    # Event = officially ungradable image (Overall quality = 0).
    true = rows["quality_good"].to_numpy(dtype=int) == 0
    gate = rows["quality_gate"].astype(str).to_numpy()
    predicted = gate != "pass" if prediction_kind == "intervention_review_or_fail" else gate == "fail"
    tp = int((predicted & true).sum())
    tn = int((~predicted & ~true).sum())
    fp = int((predicted & ~true).sum())
    fn = int((~predicted & true).sum())
    ratio = lambda a, b: float(a / b) if b else None
    recall = ratio(tp, tp + fn)
    spec = ratio(tn, tn + fp)
    return {
        "n": int(len(rows)), "true_positive_event": "official Overall quality = 0 (ungradable)",
        "sensitivity_ungradable": recall, "specificity_gradable": spec,
        "precision_ppv_ungradable": ratio(tp, tp + fp),
        "negative_predictive_value_gradable": ratio(tn, tn + fn),
        "f1_ungradable": ratio(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (recall + spec) / 2 if recall is not None and spec is not None else None,
        "under_rejection_rate": ratio(fn, tp + fn),
        "over_rejection_rate": ratio(fp, tn + fp),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def cluster_bootstrap(rows: pd.DataFrame, samples: int, seed: int) -> dict:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, patient in enumerate(rows["patient_id"].astype(str)):
        groups[patient].append(i)
    grouped = [np.asarray(indices, dtype=int) for indices in groups.values()]
    ensure(len(grouped) >= 2, "Cannot bootstrap fewer than two patients")
    rng = np.random.default_rng(seed)
    all_values: dict[str, list[float]] = defaultdict(list)
    keys = ("sensitivity_ungradable", "specificity_gradable", "balanced_accuracy")
    for _ in range(samples):
        sampled_groups = rng.integers(0, len(grouped), size=len(grouped))
        indices = np.concatenate([grouped[k] for k in sampled_groups])
        s = quality_stats(rows.iloc[indices], "intervention_review_or_fail")
        for k in keys:
            if s[k] is not None:
                all_values[k].append(s[k])
    return {
        "method": "patient-clustered nonparametric bootstrap (supplemental post-hoc quality extension)",
        "seed": seed, "samples_requested": samples, "patient_clusters": len(grouped),
        "intervals": {
            key: {"95_ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                  "valid_replicates": len(v)}
            for key, v in all_values.items() if v
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    ensure(args.bootstrap_samples >= 1, "Bootstrap samples must be >=1")
    lock, lock_file = verify_lock(project_root)
    print(f"[audit] {len(lock['files'])} frozen-file hashes checked via protocol lock.", flush=True)

    external_dir = project_root / "outputs/deepdrid_external"
    predictions_file = external_dir / "deepdrid_predictions.csv"
    metrics_file = external_dir / "deepdrid_external_metrics.json"
    exclusion_file = external_dir / "deepdrid_exclusions.csv"
    for path in (predictions_file, metrics_file, exclusion_file):
        ensure(path.is_file(), f"Missing original result: {path}")
    original_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    ensure(original_metrics["dataset"]["images"] == 1589, "Unexpected original analysis size")
    ensure(original_metrics["protocol"]["lock_sha256"].lower() == sha256(lock_file),
           "Original evaluation used a different protocol lock")

    predictions = pd.read_csv(predictions_file)
    ensure(REQUIRED_PREDICTIONS.issubset(predictions.columns), "Missing prediction columns")
    ensure(len(predictions) == 1589, "Unexpected stored prediction count")
    ensure(not predictions["image_id"].isna().any(), "Missing image IDs in predictions")
    ensure(not predictions["image_id"].duplicated().any(), "Duplicate stored prediction image ID")
    ensure(predictions["quality_gate"].isin(["pass", "review", "fail"]).all(), "Invalid stored gate")
    ensure(predictions["quality_score"].between(0, 100).all(), "Invalid stored quality score")
    ensure(predictions["quality_good"].isin([0, 1]).all(), "Invalid stored official quality label")
    ensure(predictions["patient_id"].notna().all(), "Missing patient IDs")
    ensure((predictions["patient_id"] == predictions["patient_id"].astype(int)).all(),
           "Invalid patient IDs in predictions")
    expected_quality = original_metrics["quality_gate"]
    for gate_name in ("pass", "review", "fail"):
        ensure(int((predictions["quality_gate"] == gate_name).sum()) ==
               expected_quality["gate_distribution"][gate_name],
               f"Cached {gate_name} counts disagree with original metrics")
    for mode in ("intervention_review_or_fail", "recapture_fail_only"):
        observed = quality_stats(predictions, mode)["confusion_matrix"]
        ensure(observed == expected_quality[mode]["confusion_matrix"],
               f"Cached {mode} confusion matrix disagrees with original metrics")
    ensure(int(predictions["quality_good"].sum()) == expected_quality["official_gradable"],
           "Cached quality labels disagree with original metrics")

    dataset_root = args.dataset_root.resolve()
    regular_root = dataset_root / "regular_fundus_images" if (dataset_root / "regular_fundus_images").is_dir() else dataset_root
    label_hashes = {item["partition"]: item["sha256"].lower()
                    for item in original_metrics["dataset"]["label_files"]}
    originals: dict[str, tuple[Path, dict]] = {}
    for partition, n_expected in zip(PARTITIONS, (1200, 400)):
        csv_path = regular_root / partition / f"{partition}.csv"
        ensure(csv_path.is_file(), f"DeepDRiD CSV missing: {csv_path}")
        ensure(sha256(csv_path) == label_hashes[partition],
               f"CSV SHA256 differs from original evaluation: {csv_path}")
        original = pd.read_csv(csv_path)
        ensure(REQUIRED_CSV.issubset(original.columns), f"Missing original CSV fields: {partition}")
        ensure(len(original) == n_expected, f"Unexpected {partition} record count")
        for record in original.to_dict("records"):
            image_id = normalise_identifier(record["image_id"])
            ensure(image_id not in originals, f"Duplicate image ID: {image_id}")
            ensure(value_or_none(record["Overall quality"]) in (0, 1),
                   f"Missing or invalid quality label: {image_id}")
            originals[image_id] = (csv_path, record)
    ensure(len(originals) == 1600, "Original labels do not have exactly 1600 distinct image IDs")

    pred_by_id = predictions.set_index("image_id", drop=False)
    present = set(pred_by_id.index.astype(str))
    missing = set(originals).difference(present)
    ensure(present.issubset(set(originals)), "Predictions contain unknown image IDs")
    ensure(missing == EXCLUDED, f"Unexpected unassessed image IDs: {sorted(missing ^ EXCLUDED)}")

    excluded_audit = pd.read_csv(exclusion_file)
    ensure("image_id" in excluded_audit, "Exclusion audit is missing image_id")
    ensure(set(excluded_audit["image_id"].astype(str)) == EXCLUDED and len(excluded_audit) == 11,
           "Exclusions file differs from expected 11 images")

    # Only after verifying the serialized model's hash do we load it (joblib is executable).
    sys.path.insert(0, str(project_root))
    from retina_poc.core import load_rgb, canonicalize_image, extract_features  # noqa: PLC0415
    from retina_poc.modeling import load_bundle, score_quality  # noqa: PLC0415
    bundle = load_bundle(project_root / lock["model_path"])
    print("[audit] Source CSV, model and original 1589 predictions verified.", flush=True)

    def fresh_quality(image_id: str) -> dict:
        csv_path, original = originals[image_id]
        patient_id = normalise_identifier(original["patient_id"])
        image_file = resolve_image(csv_path, patient_id, image_id, original["image_path"])
        image = canonicalize_image(load_rgb(image_file))
        _, _, quality_metrics = extract_features(image)
        return score_quality(quality_metrics, bundle["quality_reference"])

    # Verify that our current reproduction route matches two previously generated records.
    for partition in PARTITIONS:
        sample = predictions.loc[predictions["partition"] == partition].iloc[0]
        check = fresh_quality(str(sample["image_id"]))
        ensure(check["gate"] == str(sample["quality_gate"]) and
               abs(float(check["score"]) - float(sample["quality_score"])) < 1e-6,
               f"Quality reproduction mismatch: {sample['image_id']}")
    print("[audit] Training and validation reproduction checks passed.", flush=True)

    records = []
    supplemental_image_hashes = {}
    for image_id, (csv_path, original) in originals.items():
        patient_id = normalise_identifier(original["patient_id"])
        partition = csv_path.parent.name
        quality_good = value_or_none(original["Overall quality"])
        if image_id in missing:
            eye = image_id.rsplit("_", 1)[-1][0].lower()
            ensure(eye in ("l", "r"), f"Unexpected auxiliary ID: {image_id}")
            matching_grade_field = "left_eye_DR_Level" if eye == "l" else "right_eye_DR_Level"
            ensure(value_or_none(original[matching_grade_field]) is None,
                   f"Auxiliary image unexpectedly has own eye DR grade: {image_id}")
            quality = fresh_quality(image_id)
            supplemental_image_hashes[image_id] = sha256(resolve_image(
                csv_path, patient_id, image_id, original["image_path"]))
            source = "quality_only_extension_no_eye_dr_label"
            print(f"[quality-only] {image_id}: {quality['gate']} (score={quality['score']:.1f})", flush=True)
        else:
            stored = pred_by_id.loc[image_id]
            ensure(str(stored["partition"]) == partition, f"Partition disagreement: {image_id}")
            ensure(int(stored["patient_id"]) == int(patient_id), f"Patient disagreement: {image_id}")
            ensure(int(stored["quality_good"]) == quality_good, f"Quality-label disagreement: {image_id}")
            quality = {"score": float(stored["quality_score"]), "gate": str(stored["quality_gate"])}
            source = "original_1589_predictions"
        records.append({
            "partition": partition, "patient_id": patient_id, "image_id": image_id,
            "quality_good": int(quality_good), "quality_score": quality["score"],
            "quality_gate": quality["gate"], "source": source,
            "eye_dr_label_eligible": image_id not in missing,
        })

    frame = pd.DataFrame(records)
    ensure(len(frame) == 1600 and frame["image_id"].nunique() == 1600,
           "Full quality table incomplete")
    ensure(int(frame["eye_dr_label_eligible"].sum()) == 1589, "Unexpected DR eligible count")
    gate_counts = {k: int((frame["quality_gate"] == k).sum()) for k in ("pass", "review", "fail")}
    count_good = int((frame["quality_good"] == 1).sum())
    metric = {
        "schema_version": 1,
        "status": "post_hoc_quality_only_extension_to_1600_not_primary_frozen_run",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "DeepDRiD regular fundus: official training and validation CSVs",
        "images": len(frame), "patients": int(frame["patient_id"].nunique()),
        "official_gradable": count_good, "official_ungradable": int(len(frame) - count_good),
        "dr_eligible_images": 1589,
        "quality_only_extra_images": sorted(missing),
        "quality_only_extra_image_sha256": supplemental_image_hashes,
        "processing": "Reuse 1589 frozen predictions; recompute quality only for 11 extra images",
        "model_and_thresholds_modified": False,
        "quality_score_roc_auc_gradable_post_hoc": float(roc_auc_score(frame["quality_good"], frame["quality_score"])),
        "gate_distribution": gate_counts,
        "pass_coverage": gate_counts["pass"] / len(frame),
        "intervention_review_or_fail": quality_stats(frame, "intervention_review_or_fail"),
        "recapture_fail_only": quality_stats(frame, "recapture_fail_only"),
        "supplemental_cluster_bootstrap_intervention": cluster_bootstrap(
            frame, samples=args.bootstrap_samples, seed=20260823),
        "caveat": "Exploratory extension after inspecting results; not the original preregistered 1600-image run."
    }

    output = project_root / "outputs/deepdrid_quality_1600"
    output.mkdir(parents=True, exist_ok=True)
    csv_output = output / "deepdrid_quality_1600_records.csv"
    json_output = output / "deepdrid_quality_1600_metrics.json"
    manifest_output = output / "deepdrid_v05_quality_evidence_manifest.json"
    frame.to_csv(csv_output, index=False, encoding="utf-8")
    json_output.write_text(json.dumps(metric, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    inputs = [predictions_file, metrics_file, exclusion_file, lock_file,
              Path(__file__).resolve()]
    output_artifacts = [csv_output, json_output]
    file_records = []
    for path in inputs + output_artifacts:
        file_records.append({"file": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "post_hoc_supplemental_extension", "files": file_records,
                "official_label_csv_sha256": label_hashes,
                "source_protocol_sha256": sha256(lock_file)}
    manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {len(frame)} images, {len(missing)} additional quality-only images, {len(gate_counts)} gate categories", flush=True)
    print(f"[done] gradable={count_good}, ungradable={len(frame)-count_good}, patients={metric['patients']}", flush=True)
    print(json.dumps({"gate_distribution": gate_counts,
                      "intervention": metric["intervention_review_or_fail"],
                      "roc_auc": metric["quality_score_roc_auc_gradable_post_hoc"]},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"[output] {output}", flush=True)


if __name__ == "__main__":
    main()
