# Contributing to RetinaTrust

RetinaTrust is an academic research prototype. Contributions must preserve scientific traceability, conservative medical wording and a reproducible review trail.

## Branch model

- `main` is the reviewed, runnable baseline. Do not commit directly to it.
- `backend/be-<number>-<short-name>` is for API, validation and server work.
- `frontend/fe-<number>-<short-name>` is for HTML, CSS, JavaScript and accessibility work.
- `qa/qa-<number>-<short-name>` is for integration, release checks and documentation that spans components.

Create every branch from the latest `main`. Keep one task per branch and open a pull request before merge. Do not force-push `main` or rewrite reviewed history.

## Required evidence

Before requesting review, run:

```bash
python scripts/verify_artifacts.py
python -m unittest discover -s tests -v
git diff --check
```

Include the commands, results and changed-file list in the pull request. A commit hash without an accessible branch or patch is not reviewable evidence.

## Scientific change control

Changes to any of the following require a dedicated scientific task and regenerated evidence:

- `retina_poc/core.py`
- `retina_poc/modeling.py`
- `artifacts/v0.3/retina_baseline.joblib`
- fixed-test metrics or predictions
- robustness results
- classification, confidence or quality-gate policy
- `retina_poc/external_validation.py`, `scripts/evaluate_external.py` or the v0.4 protocol lock

Do not hide degraded specificity, calibration error, uncertainty or review routing. A regression comparison must pass identical image bytes through the old and new paths in the same dependency environment.

The first DeepDRiD run must remain zero-shot. Any change after external labels or metrics are inspected is a protocol deviation and requires a new version plus explicit disclosure. Never overwrite the frozen protocol lock to make a modified run appear preregistered.

## Data and privacy

- Never commit the full IDRiD dataset, local patient data or unlicensed images.
- Do not log Base64 image content, pixel data, filenames containing personal information or request bodies.
- Keep the application local by default and do not add external telemetry.
- Preserve the IDRiD citation and CC BY 4.0 attribution for the bundled demonstration image.
- Do not commit DeepDRiD images or labels; preserve the official repository's CC BY-SA 4.0 terms.

## Code and interface expectations

- Use clear, narrowly scoped commits such as `fix(api): reject oversized body before read`.
- Add tests for new behavior and for the failure mode being fixed.
- Preserve keyboard access, visible focus, reduced-motion support and non-colour-only status communication.
- Keep the disclaimer visible: this is a research prototype, not a medical device or diagnosis.
