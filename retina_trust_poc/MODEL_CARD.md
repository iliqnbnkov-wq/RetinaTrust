# RetinaTrust v0.3 baseline model card

Protocol note: the model and empirical evidence remain v0.3. RetinaTrust v0.4 adds a frozen external-validation runner but does not change or retrain this model.

## Intended use

Research proof of concept for studying how retinal-image quality and controlled photometric perturbations affect a model for referable diabetic retinopathy (DR grade 2 or higher). It is not a diagnostic system, medical device or substitute for an ophthalmologist.

## Data and protocol status

- Dataset: IDRiD, `B. Disease Grading`, CC BY 4.0.
- Official training partition: 413 images; 259 referable and 154 non-referable.
- Official testing partition: 103 images; 64 referable and 39 non-referable.
- Target: referable DR, defined as retinopathy grade ≥ 2.
- Test images are excluded from scaling, fitting, calibration, model selection and threshold selection.
- The official test partition was already inspected through v0.1 results before the protocol was corrected. It is fixed in v0.3 but is not a pristine, never-inspected locked test set.

See `docs/DATA_PROVENANCE.md` for retrieval and checksum evidence.

## Model and preprocessing

The baseline uses 178 deterministic handcrafted features: colour distributions, luminance, gradients, radial summaries, local blocks, difference-of-Gaussian responses and technical quality indicators. A robust-scaled logistic regression is selected by five-fold cross-validation on the training partition and calibrated with sigmoid calibration.

Version 0.3 applies one shared preprocessing invariant: every training, clean-test and degraded image is converted to RGB and canonicalized to a maximum dimension of 1024 px using Pillow LANCZOS before feature extraction. This removes the clean/degraded path mismatch identified in v0.1.

The model is lightweight and local. It is not state of the art. It does not localize DR lesions; the interface visualizes local contrast and explicitly does not call it Grad-CAM.

## Fixed-test results

| Metric | Result |
|---|---:|
| AUROC | 0.720 |
| Sensitivity | 0.859 |
| Specificity | 0.359 |
| F1 | 0.764 |
| Macro-F1 | 0.608 |
| Balanced accuracy | 0.609 |
| Brier score | 0.207 |
| Negative log-likelihood | 0.591 |
| Expected calibration error | 0.087 |

Bootstrap 95% CI: AUROC 0.619–0.812; Macro-F1 0.505–0.701; balanced accuracy 0.526–0.697.

At the fixed 0.50 threshold, the confusion matrix is TN 14, FP 25, FN 9, TP 55. The weak specificity is a central limitation, not a hidden result.

## Robustness evidence

Each perturbation is paired to the same clean image and evaluated through the shared canonical path. Two thousand paired bootstrap samples estimate confidence intervals for metric differences and case-level changes.

- Severe underexposure: AUROC 0.546; ΔAUROC −0.174 (95% CI −0.287 to −0.056); prediction flips 77.7% (68.9–85.4%).
- Severe gamma deviation: AUROC 0.602; ΔAUROC −0.118 (95% CI −0.213 to −0.022); prediction flips 40.8% (31.1–50.5%).
- Severe colour shift, blur and JPEG have AUROC-difference intervals that include zero in this sample.
- The v0.1 resampling-only control causes 6.8% flips (2.9–11.7%) and ΔAUROC −0.006 (−0.030 to +0.018).

The corrected analysis supports an association within this fixed synthetic experiment. It does not establish a universal camera-independent or clinical causal effect.

## Review and quality policy

- Quality factors: brightness, contrast, sharpness, retinal field coverage, illumination uniformity, colour balance and clipping.
- Confidence floor: 0.85 for the displayed provisional-result path.
- Low-quality cases are marked for repeat acquisition.
- Borderline quality or insufficient confidence is routed to human review.

The quality thresholds and confidence floor are design safeguards. They are not clinically validated operating points.

## Known limitations

- Small, single-source dataset with limited camera and population diversity.
- Official testing data reused after v0.1 protocol review; external validation is required.
- No completed evaluation on an independent centre or external dataset.
- Handcrafted features are not a competitive lesion-aware retinal architecture.
- Synthetic perturbations do not reproduce every real acquisition failure.
- No clinically annotated gradability ground truth.
- Quality scoring is relative to the IDRiD development distribution.
- The thresholded flip rate depends on the fixed 0.50 operating point.
- Bootstrap intervals quantify sampling variability on these 103 cases, not dataset shift or clinical uncertainty.
- The system does not identify patients, fuse repeated images or validate that a retake is clinically comparable.

## v0.4 external-validation status

A zero-shot DeepDRiD protocol is frozen in `docs/EXTERNAL_VALIDATION_PROTOCOL_BG.md` and `artifacts/v0.4/external_protocol_lock.json`. It uses the unchanged model, classification threshold 0.50, confidence floor 0.85 and patient-clustered bootstrap intervals.

The primary endpoint remains binary referable DR at image level. Quality-gate performance is evaluated separately against DeepDRiD `Overall quality`. Eye- and patient-level max aggregation are secondary analyses. No five-class classifier, demographic-bias isolation or universal clinical AUROC threshold is claimed.

No external result is included in this model card until the official data are processed, denominators are reviewed and the generated result files receive a new evidence manifest.
