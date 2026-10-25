from pathlib import Path
import sys
import runpy
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import retina_poc.external_validation as ev

EXCLUDED = {
    "56_r3", "56_l3",
    "77_r4", "77_r3", "77_l3", "77_l4",
    "164_r3", "164_r4", "164_l3",
    "167_r3", "167_l3",
}

original_read_csv = pd.read_csv

def audited_read_csv(path, *args, **kwargs):
    frame = original_read_csv(path, *args, **kwargs)

    if Path(path).name != "regular-fundus-training.csv":
        return frame

    if len(frame) != 1200:
        raise RuntimeError("Unexpected training CSV size.")

    mask = frame["image_id"].astype(str).isin(EXCLUDED)
    found = set(frame.loc[mask, "image_id"].astype(str))

    if found != EXCLUDED:
        raise RuntimeError(
            f"Unexpected exclusion set: {found ^ EXCLUDED}"
        )

    for _, row in frame.loc[mask].iterrows():
        image_id = str(row["image_id"])
        eye = image_id.split("_")[-1][0].lower()
        column = (
            "left_eye_DR_Level" if eye == "l"
            else "right_eye_DR_Level"
        )

        if pd.notna(row[column]):
            raise RuntimeError(
                f"{image_id} has an eye-specific DR grade. "
                "Exclusion must be reviewed."
            )

    audit_dir = ROOT / "outputs" / "deepdrid_external"
    audit_dir.mkdir(parents=True, exist_ok=True)

    audit = frame.loc[
        mask, ["patient_id", "image_id", "Overall quality"]
    ].copy()
    audit["reason"] = "Missing eye-specific DR grade"
    audit.to_csv(
        audit_dir / "deepdrid_exclusions.csv",
        index=False
    )

    cleaned = frame.loc[~mask].copy()

    print(
        f"[DeepDRiD] Excluded {mask.sum()} ungraded images.",
        flush=True
    )
    print(
        f"[DeepDRiD] Training records: {len(cleaned)}",
        flush=True
    )

    return cleaned

ev.pd.read_csv = audited_read_csv

print("[DeepDRiD] Audited compatibility loader ready.", flush=True)

runpy.run_path(
    str(ROOT / "scripts" / "evaluate_external.py"),
    run_name="__main__",
)
