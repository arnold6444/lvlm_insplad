# GOAL — InsPLAD-QA

## Objective

Build a reproducible InsPLAD QA benchmark, inspired by MMAD-style evaluation, for:

1. general anomaly detection with zero-shot vs one-shot (normal reference) comparison,
2. structured fault understanding with anomaly + defect prediction, and
3. a fair multi-model LVLM comparison under one QA/prompt/scoring contract.

## Status (2026-10-24): done

The scope was frozen on 2026-10-23 and every planned experiment is finished. Delivered:

- Hash-aligned benchmark (Tracks A/B, Aligned Core 1,378) and a frozen, manually reviewed gold v2 set (246 images, 534 questions) with a leakage-checked dev set (240 images).
- Image-only, reference and vision-expert baselines for four open LVLMs and one API pipeline on gold v2.
- A question-independent verification agent chosen on dev and run once on gold, plus an equal-call control and a greedy MiniCPM re-run.
- Dev-only follow-ups on missed missing caps (negative results in the paper appendix).
- Paper draft in English and Korean: [`reports/paper_draft_2026-10-23/`](reports/paper_draft_2026-10-23/README.md).

Remaining (paper only): compile in Overleaf, fix the title and author list, and confirm the InsPLAD volume/pages in `refs.bib`.

## Verified source facts

- `unsupervised_anomaly_detection.zip`: 17 assets, `train/test`, 27,620 JPGs.
- `supervised_fault_classification.zip`: 5 assets, `train/val`, 11,525 JPGs, no test split.
- Exact SHA-256 overlap across the two ZIPs: 8,990 images.
- Exact unsupervised-test ↔ supervised-val overlap: 1,384 images; six of them have conflicting binary GT.
- Clean Aligned Core: 1,378 identical images with matching binary GT.
- Supervised annotation schema: `<relative_image_path> <integer_class_id>`; the IDs are asset-local class indices, not binary anomaly labels.

## Benchmark tracks

- **Track A (general anomaly detection):** all 17 unsupervised assets, official `test` (6,481). Zero-shot and one-shot use the same test images and GT. The one-shot reference is a same-asset train normal (seed 42), excluding candidates whose fingerprint appears in test. Per-asset metrics report class support, because 3 assets are normal-only in test.
- **Track B (structured fault understanding):** all 5 supervised assets, official `val` (6,417). Output both `anomaly` and `defect_class`. `good`/`normal` → `normal`; any defect → `anomalous`; `corrosão` → `corrosao` without merging into `rust`. Joint Accuracy is computed only within this track.
- **Aligned Core:** `unsupervised test ∩ supervised val ∩ exact SHA-256 ∩ binary GT agreement`. The 6 conflicting images are excluded from cross-track comparisons.

## Execution policy

- One shared QA, prompt, output schema and evaluator for every model.
- Open models: `Qwen/Qwen3.5-4B`, `OpenGVLab/InternVL3_5-8B`, `openbmb/MiniCPM-V-4_5`, `llava-hf/llava-onevision-qwen2-7b-ov-hf`, unquantized, by direct Hugging Face inference on an L4/A100-class GPU. Below 20 GB VRAM the HF runners refuse to run.
- API: the Luxia two-stage pipeline (image-only condition).
- Every run records model ID, backend, benchmark commit, GPU/VRAM, protocol and status.

## Not done (out of scope)

Full Track A/B inference for every model (the runners exist but were not run for the paper), GPT/Gemini baselines, natural-language defect descriptions, Q3 location accuracy (no positive location GT), LVLM fine-tuning and LVLM judges.
