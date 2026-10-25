import re
import sys
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

print("[DeepDRiD] Applying documented parser compatibility amendment.", flush=True)

import retina_poc.external_validation as ev

# Accept images 1-4 for both eyes.
# The frozen source files and model remain unchanged.
ev._EYE_PATTERN = re.compile(
    r"_(?P<eye>[lr])[1-4]$",
    re.IGNORECASE,
)

print("[DeepDRiD] Eye identifier parser ready.", flush=True)

runpy.run_path(
    str(ROOT / "scripts" / "evaluate_external.py"),
    run_name="__main__",
)
