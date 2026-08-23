#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "artifacts/v0.4/integrity_manifest.json"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the RetinaTrust scientific artifact manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    try:
        manifest_path.relative_to(ROOT)
    except ValueError as exc:
        raise SystemExit("Manifest must be inside the project root.") from exc

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read manifest: {exc}") from exc

    if manifest.get("schema_version") != 1 or manifest.get("algorithm") != "sha256":
        raise SystemExit("Unsupported integrity-manifest schema or hash algorithm.")

    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise SystemExit("Integrity manifest contains no files.")

    failures: list[str] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            failures.append(f"Entry {index}: expected an object.")
            continue

        relative = entry.get("path")
        expected_size = entry.get("size_bytes")
        expected_hash = entry.get("sha256")
        if not isinstance(relative, str) or not relative:
            failures.append(f"Entry {index}: invalid path.")
            continue
        if relative in seen:
            failures.append(f"{relative}: duplicate manifest entry.")
            continue
        seen.add(relative)

        candidate = (ROOT / relative).resolve()
        try:
            candidate.relative_to(ROOT)
        except ValueError:
            failures.append(f"{relative}: path escapes project root.")
            continue
        if not candidate.is_file():
            failures.append(f"{relative}: missing file.")
            continue
        if type(expected_size) is not int or expected_size < 0:
            failures.append(f"{relative}: invalid expected size.")
            continue
        if not isinstance(expected_hash, str) or not SHA256_PATTERN.fullmatch(expected_hash):
            failures.append(f"{relative}: invalid expected SHA-256.")
            continue

        actual_size = candidate.stat().st_size
        actual_hash = sha256(candidate)
        if actual_size != expected_size:
            failures.append(f"{relative}: size {actual_size}, expected {expected_size}.")
        if actual_hash != expected_hash:
            failures.append(f"{relative}: SHA-256 mismatch.")
        if actual_size == expected_size and actual_hash == expected_hash:
            print(f"OK  {relative}")

    if failures:
        for failure in failures:
            print(f"FAIL  {failure}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Verified {len(entries)} integrity records.")


if __name__ == "__main__":
    main()
