# RetinaTrust v0.5 — Final Research Status

## Status
External evaluation completed. Research proof of concept.
Not validated for clinical use.

## Diagnostic evaluation
- Model: frozen RetinaTrust v0.3, trained on IDRiD.
- Internal IDRiD AUROC: approximately 0.720.
- External dataset: DeepDRiD regular fundus.
- Diagnostic evaluation: 1589 images, 399 patients.
- External AUROC: 0.462.
- Patient-clustered 95% CI: 0.422–0.502.
- Sensitivity: 72.35%.
- Specificity: 22.90%.

## Image quality evaluation
- Dataset: DeepDRiD regular fundus.
- Total: 1600 images, 400 patients.
- Officially gradable: 758.
- Officially ungradable: 842.
- Quality-score AUROC: 0.623.
- PASS: 305 images.
- REVIEW: 1101 images.
- FAIL: 194 images.
- Sensitivity for detecting ungradable images: 86.34%.
- Specificity for retaining gradable images: 25.07%.
- Ungradable images incorrectly passed: 115.
- Gradable images unnecessarily flagged: 568.

## Protocol and scientific limitations
- The original model and diagnostic thresholds were not modified.
- Eleven additional images lacked corresponding DR labels.
- Those images were excluded from diagnostic evaluation.
- They were included in the supplemental quality-only analysis.
- The 1600-image quality extension is post-hoc.
- External diagnostic performance shows substantial domain shift.
- Confidence estimates do not establish diagnostic reliability.
- Current quality gating produces excessive false interventions.
- No clinical safety or effectiveness claim is justified.

## Evidence
See RETINATRUST_V05_EVIDENCE_REPORT_BG.md.
See docs/DEEPDRID_PARSER_ADDENDUM.md.
See docs/DEEPDRID_UNGRADED_EXCLUSIONS.md.

## Next research phase
RetinaTrust v0.6:
1. Failure-mode and domain-shift analysis.
2. Improved image-quality assessment.
3. New diagnostic baseline.
4. Patient-level data separation.
5. Independent evaluation on unseen data.
