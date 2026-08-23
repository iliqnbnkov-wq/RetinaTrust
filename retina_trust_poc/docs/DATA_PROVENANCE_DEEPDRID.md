# DeepDRiD provenance plan for RetinaTrust v0.4

## Source identity

- Dataset: Deep Diabetic Retinopathy Image Dataset (DeepDRiD), regular-fundus component.
- Official repository: <https://github.com/deepdrdoc/DeepDRiD>.
- Challenge page: <https://biomedicalimaging.org/2020/wp-content/uploads/static-html-to-wp/data/dff0d41695bbae509355435cd32ecf5d/index-29.htm>.
- Paper: Liu R, Wang X, Wu Q, et al. *DeepDRiD: Diabetic Retinopathy—Grading and Image Quality Estimation Challenge.* Patterns. 2022;3(6):100512. <https://doi.org/10.1016/j.patter.2022.100512>.
- License declared by the official repository: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0). Users must inspect and comply with the repository `LICENSE` at retrieval time.

## Planned partitions

The frozen primary run uses:

- `regular_fundus_images/regular-fundus-training/regular-fundus-training.csv`;
- `regular_fundus_images/regular-fundus-validation/regular-fundus-validation.csv`;
- the corresponding sibling `Images/` directories.

The online challenge evaluation fold is not silently merged because its annotations are distributed in a different format and had a different challenge role.

## Label schema used

The parser requires the official columns:

```text
patient_id, image_id, image_path, Overall quality,
left_eye_DR_Level, right_eye_DR_Level, patient_DR_Level,
Clarity, Field definition, Artifact
```

`Overall quality` is mapped as 1 = good enough and 0 = not good enough. Eye DR grades 0–4 are mapped to the binary target `grade >= 2`. Grade 5, if present, is treated as ungradable and excluded only from DR metrics.

## Retrieval and release status

No DeepDRiD images or label files are included in the v0.4 protocol-ready archive. Therefore this file does not fabricate a retrieval date, local archive checksum or observed image count.

At execution time the runner records the exact label filenames, sizes and SHA-256 hashes in `deepdrid_external_metrics.json`. Before external results are published, the release record must also document:

- retrieval date and source commit;
- verified image and patient counts;
- missing/duplicate check result;
- license snapshot;
- result-file hashes.

Raw DeepDRiD data remains under `data/`, `dataset/` or `datasets/`, all excluded from version control.
