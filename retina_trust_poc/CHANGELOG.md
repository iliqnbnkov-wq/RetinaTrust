# Changelog

All notable project changes are recorded here. The format follows Keep a Changelog and the project uses semantic versioning for tagged releases.

## Unreleased

No unreleased changes recorded.

## 0.4.0 - 2026-08-23

### Added

- Frozen zero-shot external-validation protocol for DeepDRiD regular fundus.
- Strict parser for official training and validation CSV schemas, eye labels and image layout.
- Patient-clustered bootstrap intervals for dependent dual-view observations.
- Separate quality-gate evaluation with under-rejection, over-rejection, PPV, NPV, F1 and balanced accuracy.
- Primary image-level, selective-coverage and secondary eye/patient aggregation outputs.
- Protocol-lock verification for the unchanged v0.3 model, thresholds and scientific code paths.
- Seven external-validation unit tests and a documented audit of the Gemini proposal.

### Scientific status

- No external result is claimed in v0.4; only the pre-result protocol and executable runner are released.
- DeepDRiD replaces Messidor-2 as the primary planned dataset because it combines DR grades, image-quality labels and patient IDs.
- Cross-dataset performance is explicitly not described as an isolated demographic-bias test or clinical validation.

## 0.3.0 - 2026-08-23

### Added

- Verified IDRiD data provenance with archive hashes, file counts and license record.
- Shared canonical 1024 px LANCZOS preprocessing for training, clean and degraded paths.
- Resampling-only control for the historical v0.1 protocol mismatch.
- Per-image robustness predictions for all 103 images and 16 conditions.
- Two-thousand-sample paired bootstrap confidence intervals for metric differences and flip rates.
- Protocol regression tests for zero-difference and known-flip cases.

### Changed

- Retrained and calibrated the v0.3 baseline on the corrected training path.
- Application and interface now load versioned v0.3 artifacts.
- Stress table now exposes paired ΔAUROC and confidence intervals.
- Scientific wording now calls the official test partition fixed and reused after protocol correction, not pristine or never inspected.
- Model card, experiment report, defense brief and start guide now report v0.3 evidence.

### Scientific result

- Clean AUROC 0.720 (95% CI 0.619–0.812), Macro-F1 0.608 and ECE 0.087.
- Severe underexposure ΔAUROC −0.174 (paired 95% CI −0.287 to −0.056), with 77.7% flips.
- Legacy resampling-only ΔAUROC −0.006 (−0.030 to +0.018), with 6.8% flips.

## 0.2.0 - 2026-08-23

### Added

- GitHub Actions verification for Python 3.12.
- Pull-request template, contribution rules, architecture and release QA documentation.
- Artifact integrity manifest and verification command.
- Strict JSON, Data URL, MIME, geometry and decompression-bomb validation.
- Full local HTTP integration regression suite.
- Explicit repeat-acquisition workflow and visible evidence context.
- Scientific-positioning and Bulgarian defense notes.

### Changed

- Launchers use an isolated `.venv` and pinned dependencies.
- v0.1 robustness results were explicitly labelled preliminary after discovery of the path mismatch.

## 0.1.0 - 2026-08-23

### Added

- Local browser interface for retinal image analysis.
- Handcrafted-feature logistic-regression baseline for referable DR.
- Explicit quality and confidence gates.
- Controlled perturbations and initial robustness matrix.
- IDRiD test metrics, model card and experiment report.
- Attributed demonstration image, automated core tests and launch scripts.
