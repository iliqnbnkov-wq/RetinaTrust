#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_MANIFEST = ROOT / "artifacts/v0.3/integrity_manifest.json"
DEFAULT_OUTPUT = ROOT / "artifacts/v0.4/integrity_manifest.json"

V04_FILES = {
    "artifacts/v0.3/integrity_manifest.json": "Historical v0.3 integrity record",
    "artifacts/v0.4/external_protocol_lock.json": "Frozen pre-result external protocol",
    "retina_poc/external_validation.py": "DeepDRiD parser, metrics and clustered evaluation",
    "scripts/evaluate_external.py": "Frozen zero-shot external runner",
    "scripts/verify_artifacts.py": "Integrity-verification command",
    "scripts/build_integrity_manifest.py": "Reproducible v0.4 manifest builder",
    "tests/test_external_validation.py": "External-protocol regression tests",
    "requirements.lock": "Pinned dependency environment",
    ".github/workflows/ci.yml": "Automated compile, integrity and test workflow",
    "README.md": "Release instructions and scientific status",
    "MODEL_CARD.md": "Model evidence and external-validation status",
    "CHANGELOG.md": "Versioned change record",
    "CONTRIBUTING.md": "Scientific change-control policy",
    "START_HERE_BG.md": "Bulgarian start and presentation guide",
    "THIRD_PARTY_NOTICES.md": "Dataset and dependency notices",
    "docs/ARCHITECTURE.md": "System and external-run boundaries",
    "docs/QA_CHECKLIST.md": "Release and external-protocol checks",
    "docs/SCIENTIFIC_POSITIONING_BG.md": "Conservative scientific positioning",
    "docs/DEFENSE_BRIEF_BG.md": "Bulgarian defense brief",
    "docs/EXTERNAL_VALIDATION_PROTOCOL_BG.md": "Human-readable frozen protocol",
    "docs/DATA_PROVENANCE_DEEPDRID.md": "External-dataset provenance plan",
    "docs/GEMINI_AUDIT_BG.md": "Source-checked audit of the delegated proposal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the RetinaTrust v0.4 integrity manifest.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(relative: str, role: str) -> dict[str, object]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": relative,
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256(path),
        "role": role,
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    try:
        output.relative_to(ROOT)
    except ValueError as exc:
        raise SystemExit("Output manifest must be inside the project root.") from exc

    historical = json.loads(HISTORICAL_MANIFEST.read_text(encoding="utf-8"))
    files = [
        record(entry["path"], f"v0.3 evidence — {entry['role']}")
        for entry in historical["files"]
    ]
    existing = {entry["path"] for entry in files}
    for relative, role in V04_FILES.items():
        if relative in existing:
            continue
        files.append(record(relative, role))

    payload = {
        "schema_version": 1,
        "algorithm": "sha256",
        "release": "0.4.0",
        "status": "protocol_ready_external_results_not_run",
        "description": (
            "Integrity record for the verified v0.3 empirical evidence and the v0.4 "
            "frozen DeepDRiD external-validation protocol."
        ),
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(files)} records to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
