# GOAL — InsPLAD LVLM Benchmark v1

## Objective
Build a reproducible InsPLAD QA benchmark inspired by MMAD's QA evaluation style for anomaly detection and structured explainability.

## Verified source facts
- Unsupervised: 17 asset categories, `train/test`, 27,620 JPG images.
- Supervised: 5 asset categories, `train/val`, 11,525 JPG images, no test split.
- Benchmark Core: all 5 shared asset categories.
- `good` and `normal` are normalized to benchmark `normal` while raw labels are preserved.
- `corrosão` is mechanically normalized to `corrosao`; it is not merged with `rust`.
- Structured explainability uses supervised `val` because no `test` exists.
- One-shot uses one same-asset train-normal reference per asset, seed 42, excluding test-duplicate fingerprints.

## Stage deliverables
- Dataset structure/count/intersection reports.
- Zero-shot and one-shot anomaly QA generator.
- Structured anomaly+defect QA generator.
- QA validation and leakage checks.
- Balanced Accuracy/F1/Macro F1/Joint Accuracy/FDR/Consistency/Invalid metrics and confusion matrix.
- Deterministic 20-sample review sets.

## Out of scope
Natural-language defect descriptions, defect localization, Object IoU, LVLM Judge, fine-tuning, new model development, performance-improvement methods, and large-scale LVLM inference.

## Remaining source checks
The connected Drive interface cannot raw-download the 875 MB/1.17 GB ZIPs. Exact SHA256 overlap and the contents of the five `image_list_val.txt` files must therefore be read locally from the source ZIPs with the provided scripts.
