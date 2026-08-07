# Goal — InsPLAD LVLM Benchmark v1

## Objective
Build an InsPLAD-based LVLM benchmark for anomaly detection and structured explainability, inspired by MMAD's QA evaluation style.

## In scope
- Verify the real source ZIP structures before deriving categories/splits/labels.
- Generate anomaly zero-shot / one-shot QA from the unsupervised fault dataset.
- Generate structured anomaly + defect QA from the supervised fault dataset.
- Validate QA for leakage, duplicates, label/category consistency, and malformed JSONL.
- Evaluate anomaly and structured-explainability metrics overall and by asset.

## Out of scope
Natural-language defect descriptions, localization/IoU, LVLM Judge, fine-tuning, new model development, performance-improvement methods, and large-scale LVLM inference.

## Source of truth
The two Google Drive archives linked by `insplad_google_drive` on `main` are authoritative. Dataset-specific code must not infer their internal schema from papers or READMEs.

## Current blocker
The connected Drive interface identifies both archives, but rejects raw downloads above 100 MB. Their internal members therefore remain unverified in this environment. Dataset-specific QA generation is intentionally not implemented until a file manifest or extracted structure from these exact archives is available.
