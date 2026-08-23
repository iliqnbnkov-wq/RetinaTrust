# Controlled robustness experiment — v0.3

## Evidence status

Version 0.3 repairs the main scientific confound in v0.1. Training images, clean test images and every degraded branch now use the same RGB and max-dimension 1024 px LANCZOS canonicalization before feature extraction. The experiment stores per-image probabilities and calculates 2,000-sample paired bootstrap confidence intervals.

The official IDRiD test partition is fixed, but it was reused after the v0.1 results exposed the protocol flaw. It is therefore not described as a pristine locked test set. Its labels were not used for v0.3 fitting, calibration, model selection or threshold selection.

## Protocol

- Dataset: official IDRiD testing partition, 103 images.
- Target: referable diabetic retinopathy, DR grade ≥ 2.
- Fixed v0.3 model and fixed 0.50 classification threshold.
- Perturbations: underexposure, gamma deviation, colour shift, Gaussian blur and JPEG compression.
- Three predefined severity levels per perturbation.
- Paired design: every degraded image derives from the same clean test image.
- Shared canonical path: RGB, maximum dimension 1024 px, Pillow LANCZOS.
- Clean prediction reproduction error: at most 1.11×10⁻¹⁶.
- Bootstrap: 2,000 paired samples with a fixed random seed.

## Clean result

| Metric | Result | Bootstrap 95% CI |
|---|---:|---:|
| AUROC | 0.720 | 0.619–0.812 |
| Macro-F1 | 0.608 | 0.505–0.701 |
| Balanced accuracy | 0.609 | 0.526–0.697 |
| Sensitivity | 0.859 | — |
| Specificity | 0.359 | — |
| ECE | 0.087 | — |

## Severity level 3

| Condition | AUROC | ΔAUROC (95% CI) | Macro-F1 | Prediction flips (95% CI) | Quality intervention |
|---|---:|---:|---:|---:|---:|
| Clean | 0.720 | 0.000 | 0.608 | 0.0% | 25.2% |
| Underexposure | 0.546 | −0.174 (−0.287 to −0.056) | 0.275 | 77.7% (68.9–85.4%) | 100.0% |
| Gamma deviation | 0.602 | −0.118 (−0.213 to −0.022) | 0.550 | 40.8% (31.1–50.5%) | 100.0% |
| Colour shift | 0.657 | −0.063 (−0.181 to +0.054) | 0.518 | 20.4% (12.6–28.2%) | 86.4% |
| Blur | 0.666 | −0.054 (−0.130 to +0.021) | 0.630 | 36.9% (28.2–46.6%) | 96.1% |
| JPEG compression | 0.687 | −0.033 (−0.136 to +0.075) | 0.658 | 41.7% (33.0–51.5%) | 80.6% |

Quality intervention is the proportion routed either to repeat acquisition or human review. The 25.2% clean intervention rate shows that the hand-designed gate is conservative even without synthetic degradation.

## Interpretation

Severe underexposure produces the clearest observed failure: AUROC falls by 0.174, its paired interval remains below zero, Macro-F1 falls by 0.333, and 77.7% of thresholded decisions change. Severe gamma deviation also has a paired AUROC interval below zero.

The colour-shift, blur and JPEG point estimates are lower than clean AUROC, but their intervals include zero. Their AUROC effects are unresolved on 103 cases. They still create substantial case-level instability. A prediction flip is not automatically a new error: it records thresholded disagreement with the clean model output, so aggregate threshold metrics may occasionally improve while individual decisions change.

The quality gate intervenes in most severe synthetic failures, but it is hand-designed and relative to IDRiD. This is workflow evidence, not a validated gradability or safety claim.

## v0.1 resampling-only control

The legacy v0.1 model was re-evaluated on the same original test images after only the old 1024 px resampling transform, with no named corruption.

| Comparison | Direct clean path | Resampling-only path | Paired change (95% CI) |
|---|---:|---:|---:|
| AUROC | 0.732 | 0.727 | −0.006 (−0.030 to +0.018) |
| Macro-F1 | 0.607 | 0.600 | −0.006 (−0.075 to +0.054) |
| Prediction flips | — | 6.8% | 2.9–11.7% |
| Mean absolute probability shift | — | 0.030 | 0.024–0.037 |

This confirms that the original path mismatch caused measurable case-level instability. It does not support a resolved aggregate AUROC loss from resampling alone, and it is far too small to explain the corrected 77.7% severe-underexposure flip rate. The v0.1 results remain historical and should not be mixed with v0.3 outputs.

## Remaining validity limits

- single public dataset and small test sample;
- official test partition reused after protocol review;
- synthetic rather than naturally annotated acquisition failures;
- no external camera, centre or population validation;
- no clinical gradability labels;
- one lightweight baseline and one fixed decision threshold;
- no correction for testing many perturbation-level outcomes.

The results are reproducible PoC evidence. They are not clinical validation and should not be generalized to all retinal cameras, populations or models.

## Evidence files

- `artifacts/v0.3/locked_test_metrics.json`
- `artifacts/v0.3/locked_test_predictions.csv`
- `artifacts/v0.3/robustness_results.json`
- `artifacts/v0.3/robustness_results.csv`
- `artifacts/v0.3/robustness_predictions_long.csv`
- `artifacts/v0.3/legacy_resampling_control.json`
- `artifacts/v0.3/legacy_resampling_control_predictions.csv`
- `scripts/evaluate_robustness.py`
