# DeepDRiD External Validation — Data Exclusion Addendum

Original protocol: v0.4.0
Model: RetinaTrust v0.3
Reason: Eleven extra fundus views lack eye-specific DR grades.

Amendment:
Exclude exactly 11 specified training records before evaluation.
Do not infer missing grades from opposite-eye columns.

Expected included records: 1589.
Full 1600-image quality-only analysis: pending separately.

Original dataset files: unchanged.
Frozen model: unchanged.
Thresholds: unchanged.
Training and recalibration: none.

This is a documented deviation from the original
pre-result dataset inclusion protocol.

Audited adapter SHA256: 9D5D1E693C4F7D5DC557B06E36C9DA0ECE913F18751CE0324485DACD5079E07A
