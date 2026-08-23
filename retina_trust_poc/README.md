# RetinaTrust — quality-aware retinal AI PoC

RetinaTrust is a local research proof of concept for studying how retinal-image quality and controlled acquisition failures affect the reliability of an AI baseline. It combines a reproducible Python experiment with a browser interface.

Release status: v0.4 adds a frozen, testable DeepDRiD external-validation protocol. The bundled model and all reported empirical metrics remain the verified v0.3 evidence; **no external performance result is claimed yet**.

The application:

1. reports technical quality factors separately;
2. runs a calibrated baseline for referable diabetic retinopathy (DR grade ≥ 2);
3. separates probability, confidence and the workflow decision;
4. routes low-quality or uncertain inputs to repeat acquisition or human review;
5. runs controlled stress tests for exposure, gamma, colour balance, blur and JPEG compression.

This is not a medical device and its output is not a diagnosis. The project does not claim that image-quality assessment, uncertainty-aware referral or repeat acquisition are new ideas. Its contribution is a small, reproducible integration and controlled demonstration. See [Scientific positioning (BG)](docs/SCIENTIFIC_POSITIONING_BG.md) and [Defense brief (BG)](docs/DEFENSE_BRIEF_BG.md).

## Fast start

Windows:

1. Extract the project folder.
2. Double-click `start_retinatrust.bat`.
3. On first start, allow Python to create `.venv` and install the pinned packages.
4. Open `http://127.0.0.1:8765` if the browser does not open automatically.

Manual Windows start:

```powershell
py -3 -m pip install -r requirements.lock
py -3 app.py
```

Linux/macOS:

```bash
chmod +x start_retinatrust.sh
./start_retinatrust.sh
```

## Verify the checkout

```bash
python scripts/verify_artifacts.py
python -m unittest discover -s tests -v
```

The integrity command verifies the versioned model and scientific outputs against SHA-256 records. Automated tests cover the image pipeline, paired analysis and local HTTP boundary.

## Demonstration flow

1. Select **Load demonstration IDRiD image**.
2. Run the clean input and record quality, DR probability, confidence and workflow decision.
3. Apply underexposure or blur at severity 1, 2 and 3.
4. Observe when the result is routed to human review or repeat acquisition.
5. Use the level-3 stress table to distinguish aggregate performance change from case-level prediction flips.

The defensible interpretation is: **the prototype makes model output conditional on measurable input quality and predictive confidence**. It does not establish clinical safety or diagnostic validity.

## Reproduce v0.3

Obtain and extract IDRiD `B. Disease Grading`, then run:

```bash
python scripts/train_baseline.py --data-root "/path/to/B. Disease Grading" --force-features
python scripts/evaluate_robustness.py --data-root "/path/to/B. Disease Grading"
```

The training script uses only the 413-image official training partition for feature scaling, model selection, calibration and review-policy selection. The 103-image official testing partition is fixed, but it was reused after review of the flawed v0.1 protocol; it must not be described as a pristine, never-inspected locked test set. Test labels are not used for fitting or threshold selection in v0.3.

All training, clean and degraded inputs first pass through the same max-dimension 1024 px LANCZOS canonicalization. The robustness script stores per-image probabilities and computes 2,000-sample paired bootstrap confidence intervals.

## Frozen v0.4 external-validation protocol

DeepDRiD, not Messidor-2, is the primary external dataset. The official DeepDRiD regular-fundus labels combine patient IDs, eye/patient DR grades and image-quality annotations. The official Messidor-2 distribution does not itself include DR ground truth, so it cannot support the same pre-specified joint evaluation.

The v0.4 runner:

- verifies model, code and threshold hashes before processing data;
- uses the public DeepDRiD training and validation CSV partitions without fitting or recalibration;
- evaluates binary referable DR at image level;
- evaluates the quality gate against `Overall quality` with under- and over-rejection rates;
- computes patient-clustered bootstrap intervals because views from one patient are dependent;
- reports selective metrics only together with coverage;
- keeps eye- and patient-level max aggregation secondary and explicit.

After obtaining the official DeepDRiD repository:

```bash
python scripts/evaluate_external.py --dataset-root "/path/to/DeepDRiD"
```

Outputs are written to `outputs/deepdrid_external/` and are not release evidence until independently reviewed and added to a new integrity manifest. See [the frozen Bulgarian protocol](docs/EXTERNAL_VALIDATION_PROTOCOL_BG.md), [DeepDRiD provenance plan](docs/DATA_PROVENANCE_DEEPDRID.md) and [Gemini proposal audit](docs/GEMINI_AUDIT_BG.md).

## Verified v0.3 baseline

- Official fixed test split: 103 images.
- AUROC: 0.720 (bootstrap 95% CI 0.619–0.812).
- Sensitivity: 0.859.
- Specificity: 0.359.
- Macro-F1: 0.608.
- ECE: 0.087.

The low specificity is visible by design: 25 of 39 non-referable cases are false positives at the fixed 0.50 threshold. This lightweight baseline is an experimental object for failure-mode analysis, not a competitive retinal classifier.

## Verified v0.3 robustness result

At severity level 3:

| Condition | AUROC | Paired ΔAUROC (95% CI) | Prediction flips (95% CI) |
|---|---:|---:|---:|
| Underexposure | 0.546 | −0.174 (−0.287 to −0.056) | 77.7% (68.9–85.4%) |
| Gamma deviation | 0.602 | −0.118 (−0.213 to −0.022) | 40.8% (31.1–50.5%) |
| Colour shift | 0.657 | −0.063 (−0.181 to +0.054) | 20.4% (12.6–28.2%) |
| Blur | 0.666 | −0.054 (−0.130 to +0.021) | 36.9% (28.2–46.6%) |
| JPEG compression | 0.687 | −0.033 (−0.136 to +0.075) | 41.7% (33.0–51.5%) |

Severe underexposure and gamma deviation have paired AUROC intervals entirely below zero in this sample. The colour, blur and JPEG AUROC intervals include zero, so their discriminative-performance change is unresolved here even though many individual classifications flip. A flip is instability, not automatically an error or a performance loss.

The legacy v0.1 resampling-only control causes 6.8% flips (95% CI 2.9–11.7%) but only ΔAUROC −0.006 (95% CI −0.030 to +0.018). The old protocol was confounded, but resampling alone does not explain the much larger corrected underexposure effect. These results remain single-dataset PoC evidence, not universal causal estimates.

See [Controlled robustness experiment](EXPERIMENT_RESULTS.md) and the machine-readable files under `artifacts/v0.3/`.

## Project layout

```text
retina_trust_poc/
├── app.py                         local HTTP application
├── retina_poc/
│   ├── core.py                    canonicalization, features and perturbations
│   ├── modeling.py                inference, metrics and workflow policy
│   ├── validation.py              request and image validation
│   └── external_validation.py     DeepDRiD schema, metrics and clustered inference
├── scripts/
│   ├── train_baseline.py          reproducible model training
│   ├── evaluate_robustness.py     paired perturbation matrix and controls
│   ├── evaluate_external.py       frozen zero-shot DeepDRiD evaluation
│   └── verify_artifacts.py        SHA-256 evidence verification
├── artifacts/v0.3/                versioned model and scientific outputs
├── artifacts/v0.4/                protocol lock; no external results
├── static/                        browser interface
├── demo/                          attributed IDRiD demonstration image
├── docs/                          provenance, architecture and release notes
└── tests/                         automated scientific and API checks
```

## Data and licensing

The full IDRiD dataset is not tracked or included in the release archive. It is licensed CC BY 4.0. Retrieval source, archive checksum, file counts and protocol caveats are recorded in [Data provenance](docs/DATA_PROVENANCE.md).

Source paper: Porwal P, Pachade S, Kamble R, et al. *Indian Diabetic Retinopathy Image Dataset (IDRiD).* Data. 2018;3(3):25. [doi:10.3390/data3030025](https://doi.org/10.3390/data3030025).

RetinaTrust source and documentation are under the MIT license. The demonstration image remains under CC BY 4.0; see `THIRD_PARTY_NOTICES.md` and `demo/ATTRIBUTION.md`.

No DeepDRiD file is redistributed. The official DeepDRiD repository declares CC BY-SA 4.0; users must comply with its current `LICENSE` when retrieving and using the data.
