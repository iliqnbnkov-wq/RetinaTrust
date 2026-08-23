# IDRiD data provenance for RetinaTrust v0.3

## Dataset identity

- Dataset: Indian Diabetic Retinopathy Image Dataset (IDRiD), task `B. Disease Grading`.
- Creators named by the archive license: Prasanna Porwal, Samiksha Pachade and Manesh Kokare.
- Official challenge portal: <https://idrid.grand-challenge.org/Data/>.
- Dataset paper: Porwal P, Pachade S, Kamble R, et al. *Indian Diabetic Retinopathy Image Dataset (IDRiD).* Data. 2018;3(3):25. <https://doi.org/10.3390/data3030025>.
- License in the downloaded archive: Creative Commons Attribution 4.0 International, <https://creativecommons.org/licenses/by/4.0/>.

## Retrieval record

Retrieved on 2026-08-23 from the Zenodo record <https://zenodo.org/records/17219542>, DOI <https://doi.org/10.5281/zenodo.17219542>.

The Zenodo deposit is a redistribution and is not the original IDRiD creator or the authoritative challenge host. It was used because the official portal directs full-download access through IEEE DataPort. Scientific identity is cross-checked against the official portal, paper, archive structure and embedded IDRiD license. Future reproduction should prefer the official source when accessible.

Archive file: `B. Disease Grading.zip`

| Check | Verified value |
|---|---|
| Size | 212,405,123 bytes |
| MD5 | `b9239a4b956021a1cf0225522f11f58f` |
| SHA-256 | `8a9f4752b35d74cc35ff48b21ad44f295a6f800110ec218fc2d1c264803e4d8c` |
| ZIP integrity | `unzip -t` completed with no errors |
| Training JPEG files | 413 |
| Testing JPEG files | 103 |
| Ground-truth CSV files | 2 |

The MD5 equals the checksum displayed by the Zenodo record. The archive includes `LICENSE.txt` and `CC-BY-4.0.txt`.

## Labels used

The two grading CSVs contain `Image name` and `Retinopathy grade`. RetinaTrust defines the binary target as:

```text
referable DR = Retinopathy grade >= 2
```

Class counts after parsing:

| Partition | Images | Referable | Non-referable |
|---|---:|---:|---:|
| Official training | 413 | 259 | 154 |
| Official testing | 103 | 64 | 39 |

## Storage and release policy

The extracted full dataset lives under `data/IDRiD/` during local reproduction and is excluded by `.gitignore`. It is not part of the release archive. Only the already attributed demonstration image under `demo/` is distributed with the project.

## Protocol caveat

The official 103-image testing partition was used in v0.1 before the preprocessing flaw was identified. Version 0.3 reuses the same fixed partition after correcting the protocol. The test labels remain excluded from fitting, model selection, calibration and threshold selection, but the split is no longer pristine or never inspected. External validation is required before any stronger generalization.
