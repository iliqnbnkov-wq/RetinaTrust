# RetinaTrust release QA checklist

A release is accepted only when every applicable item is checked and the evidence is attached to the pull request or release record.

## 1. Repository integrity

- [ ] Work is based on the latest reviewed `main`.
- [ ] The change has a scoped branch and accessible commit or patch.
- [ ] `git diff --check` is clean.
- [ ] No raw dataset, secret, environment file, cache or generated package is tracked.
- [ ] `python scripts/verify_artifacts.py` passes against the v0.4 protocol manifest.

## 2. Automated verification

- [ ] Dependency installation succeeds from `requirements.lock`, or from `requirements.txt` before the lock file exists.
- [ ] `python -m compileall -q app.py retina_poc scripts tests` passes.
- [ ] `python -m unittest discover -s tests -v` passes.
- [ ] GitHub Actions passes on the pull request.

## 3. API and input safety

- [ ] Oversized or invalid `Content-Length` is rejected before the body is read.
- [ ] Only `application/json` is accepted by `/api/analyze`.
- [ ] Only exact JPEG and PNG Base64 Data URL prefixes are accepted.
- [ ] Declared MIME type matches the decoded image format.
- [ ] Decoded-byte, minimum-dimension and maximum-pixel limits are tested.
- [ ] Decompression-bomb warnings and errors are handled as client errors.
- [ ] EXIF orientation remains equivalent to the baseline loader.
- [ ] Logs contain endpoint, status and duration but no request body, Base64 data or image content.
- [ ] Static-file path traversal remains blocked.

## 4. Scientific regression

- [ ] Training, clean and degraded images share the canonical 1024 px LANCZOS preprocessing invariant.
- [ ] Stored clean predictions are reproduced by the robustness path within the declared numerical tolerance.
- [ ] Paired-comparison tests return exact zero for identical probabilities and the expected flip rate for a known case.
- [ ] Probability, confidence, quality score, quality gate and decision code match the v0.3 regression fixtures.
- [ ] The feature schema still matches the trained artifact.
- [ ] The clean demonstration request is not rejected for geometry.
- [ ] Extreme combined degradation is routed to repeat acquisition.

### External-validation protocol

- [ ] DeepDRiD CSV parsing rejects missing columns, duplicate image IDs and eye/grade mismatches.
- [ ] The runner verifies frozen code, model and threshold hashes before inference.
- [ ] The primary run performs no fitting, recalibration or threshold selection on DeepDRiD.
- [ ] DR grade 5, if present, remains in quality analysis and is excluded from binary DR denominators.
- [ ] Primary DR metrics are image-level and 95% intervals resample patient clusters.
- [ ] Quality-gate outputs include under- and over-rejection, not only NPV.
- [ ] Selective metrics always include coverage.
- [ ] Grade-stratified output is described as binary error analysis, not a five-class classifier.
- [ ] No external metric is added to documentation before the generated JSON/CSV receive an evidence manifest.

Two distinct reference paths currently exist:

| Path | Size | DR probability | Quality | Gate | Decision |
|---|---:|---:|---:|---|---|
| Original demo file | 1400×930 | 0.9450 | 97.0 | pass | provisional_positive |
| `/api/sample` JPEG roundtrip | 1100×731 | 0.9521 | 96.7 | pass | provisional_positive |

The difference is caused by resize and JPEG re-encoding. It is not an acceptable A/B regression tolerance argument.

## 5. Interface and accessibility

- [ ] Complete keyboard-only flow works: upload, demo, degradation, severity, normalization and analysis.
- [ ] Every interactive element has a visible focus indicator.
- [ ] Exactly one severity level exposes the selected ARIA state.
- [ ] Status is not communicated by colour alone.
- [ ] `prefers-reduced-motion` disables decorative movement and smooth scrolling.
- [ ] Normal text contrast is at least 4.5:1 on its rendered background.
- [ ] Model metrics include confidence intervals and confusion-matrix context.
- [ ] Layout is checked at 1440, 1024, 768 and 390 px without page-level horizontal overflow.
- [ ] Browser console contains no errors.

## 6. Medical wording and privacy

- [ ] The visible interface states that RetinaTrust is not a medical diagnosis or medical device.
- [ ] Probabilities are not relabelled as diagnostic certainty.
- [ ] Low specificity and single-dataset limitations remain visible.
- [ ] Uploaded images are not stored or transmitted externally.
- [ ] IDRiD attribution and CC BY 4.0 notice remain present.

## 7. Release record

- [ ] `CHANGELOG.md` describes the release.
- [ ] README quick-start commands were tested on a clean environment.
- [ ] Model card, experiment report and UI agree with the bundled v0.3 JSON/CSV evidence.
- [ ] Data provenance records source, license, archive hashes and exact file counts.
- [ ] If DeepDRiD was run, its source commit, label hashes, patient/image counts and license snapshot are recorded.
- [ ] Desktop and mobile screenshots correspond to the release commit.
- [ ] The release commit is tagged only after final QA approval.
