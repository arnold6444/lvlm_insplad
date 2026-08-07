# GOAL — InsPLAD LVLM Benchmark v1

## Objective

Build a reproducible InsPLAD QA benchmark, inspired by MMAD-style evaluation, for:

1. general anomaly detection with zero-shot vs one-shot comparison,
2. structured fault understanding with anomaly + defect prediction, and
3. a fair multi-model LVLM baseline comparison under one QA/prompt/scoring contract.

## Verified source facts

- `unsupervised_anomaly_detection.zip`: 17 assets, `train/test`, 27,620 JPGs.
- `supervised_fault_classification.zip`: 5 assets, `train/val`, 11,525 JPGs, no test split.
- Exact SHA256 overlap across the two ZIPs: 8,990 images.
- Exact unsupervised-test ↔ supervised-val overlap: 1,384 images.
- Six of those exact same images have conflicting binary GT.
- Clean Aligned Core: 1,378 exact same images with matching binary GT.
- Supervised annotation schema: `<relative_image_path> <integer_class_id>`.
- Annotation class IDs are asset-local class indices, not binary anomaly labels.

## Benchmark tracks

### Track A — General Anomaly Detection

- Use all 17 unsupervised asset categories.
- Evaluate on official `test`: 6,481 images.
- Zero-shot and one-shot use the exact same test images and GT.
- One-shot uses one same-asset `train/good` reference per asset, `seed=42`.
- Exclude train references whose content fingerprint appears in test.
- Keep per-asset metrics; explicitly report class support because 3 assets are normal-only in test.

### Track B — Structured Fault Understanding

- Use all 5 supervised asset categories.
- No supervised test split exists, so evaluate on official `val`: 6,417 images.
- Output both `anomaly` and `defect_class`.
- `good` and `normal` map to benchmark `normal`; any concrete defect maps to `anomalous`.
- `corrosão` is mechanically normalized to `corrosao`; do not merge it with `rust`.
- Joint Accuracy is computed only within this track.

### Aligned Core

Definition:

`unsupervised test ∩ supervised val ∩ exact SHA256 match ∩ binary GT agreement`

Use it to link zero-shot, one-shot, and structured results for the same image bytes. Do not use the 6 binary-GT-conflicting exact images in clean cross-track comparisons.

## Baseline execution policy

Use the same QA, prompts, output schema, and evaluator for all models, but run each family through the most reproducible practical backend:

- Closed models: official provider API.
  - OpenAI `gpt-5.6-sol` via Responses API.
  - Google `gemini-3.6-flash` via Gemini API.
- Open-source models: direct Hugging Face inference on a rented GPU.
  - Qwen3-VL 8B
  - InternVL 3.5 8B
  - MiniCPM-V 4.5
  - LLaVA-OneVision 7B

Do not silently quantize open-source baselines on low-VRAM hardware. If the runtime has less than 20 GB VRAM, skip direct HF baselines and use an L4/A100-class runtime instead. Record requested model ID, backend, benchmark commit, GPU/VRAM, and run status with every experiment.

## Stage deliverables

- Dataset structure/count/split/annotation reports.
- Track A zero-shot and one-shot QA generation.
- Track B structured QA generation.
- Exact-overlap and Aligned Core mapping.
- QA validation and leakage checks.
- Balanced Accuracy, F1, Macro F1, Joint Accuracy, FDR, Consistency Error, Invalid Output Rate, per-defect metrics, confusion matrix.
- Deterministic 20-sample review sets.
- Fixture-based automated tests.
- Hybrid API + GPU smoke baseline runner with a unified leaderboard and prediction review export.

## Out of scope for this stage

Natural-language defect descriptions, localization/Object IoU, LVLM Judge, LVLM fine-tuning, new performance-improvement methods, and full-dataset model runs before smoke-test validation.
