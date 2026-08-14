# InsPLAD LVLM Benchmark v1

InsPLAD-based benchmark for **General Anomaly Detection** and **Structured Fault Understanding**, inspired by MMAD-style LVLM evaluation. The benchmark compares multiple VLMs under the same QA, prompt, and scoring contract.

## Benchmark design

### Track A — General Anomaly Detection
Source: `unsupervised_anomaly_detection.zip`

- 17 asset categories
- official `test`: **6,481** images
- Zero-shot: target test image only
- One-shot: same target test image + one fixed same-asset **normal/good** image from `train`
- 14 assets contain both normal and anomalous test GT
- 3 assets are normal-only in test: `damper-preformed`, `spacer`, `yoke`

The benchmark's 1-shot regime is intentionally the MMAD-style **normal-reference** setting. It does not provide an anomalous/positive example.

### Track B — Structured Fault Understanding
Source: `supervised_fault_classification.zip`

- 5 asset categories
- `train/val`, no official test
- official `val`: **6,417** images
- output: `{"anomaly":"...","defect_class":"..."}`
- metrics include defect BAcc, Macro F1, Joint Accuracy, FDR, Consistency Error, and Invalid Output Rate

### Aligned Core — same-image cross-track analysis
The two ZIPs share **8,990 SHA256-identical images**. `unsupervised test ∩ supervised val` contains 1,384 exact same images; six have conflicting binary GT, leaving a clean **Aligned Core of 1,378 images**.

## Source-label policy

- `good` and `normal` → benchmark `normal`
- any defect folder → `anomalous`
- `corrosão` → `corrosao` slug only; it is not merged with `rust`
- supervised `image_list_val.txt` integers are asset-local class IDs, not binary anomaly labels, so semantic GT comes from defect folder names

## Setup / tests

```powershell
python -m pip install -e ".[dev]"
python -m pytest -v
```

## Main generated benchmark files

- `data/qa/anomaly_zero_shot.jsonl` — 6,481
- `data/qa/anomaly_one_shot.jsonl` — 6,481
- `data/qa/defect_qa.jsonl` — 6,417
- `reports/aligned_core.csv` — 1,378 same-image links
- `samples/anomaly_zero_shot_20.jsonl`
- `samples/anomaly_one_shot_20.jsonl`
- `samples/defect_20.jsonl`
- `samples/aligned_core_20.jsonl`

## Baseline policy — Hugging Face VLMs only

`notebooks/lvlm_baselines_colab.ipynb` is the single Colab execution entrypoint. It keeps one QA/prompt/evaluation contract and runs every Hugging Face VLM sequentially on the same smoke set.

OpenAI/Gemini API baselines have been removed. Luxia is deferred because its public Vision API is Document AI for document images, not a general-image VLM suitable for InsPLAD power-line photographs.

### 7-model GPU benchmark

Models are grouped so the final analysis can compare **MMAD-era anchors vs current/latest open VLMs**:

- `Qwen/Qwen2.5-VL-7B-Instruct` — MMAD-era anchor
- `Qwen/Qwen3-VL-8B-Instruct` — current Qwen
- `Qwen/Qwen3.5-4B` — newest L4-friendly Qwen baseline
- `OpenGVLab/InternVL2_5-8B` — MMAD-era anchor
- `OpenGVLab/InternVL3_5-8B` — current InternVL
- `openbmb/MiniCPM-V-4_5` — current compact VLM
- `llava-hf/llava-onevision-qwen2-7b-ov-hf` — MMAD-family baseline

`Qwen3.5-4B` is deliberately used instead of the 9B/27B variants because the paper smoke environment is an NVIDIA L4 with about 22 GB VRAM. The benchmark does not mix in 4-bit quantization only for one model, because that would add a separate quantization variable to the model comparison.

Qwen3.5 support requires the Transformers v5 generation; the Colab notebook pins `transformers==5.14.0` for reproducibility.

### GPU-aware adapter policy

- Qwen2.5/Qwen3/Qwen3.5 use the generic Transformers image-text pipeline.
- MiniCPM-V 4.5 uses its official `AutoModel + chat()` path.
- LLaVA-OneVision uses the official `AutoProcessor + AutoModelForMultimodalLM` path instead of the previously failing generic runner.
- InternVL2.5/3.5 keep their official `chat()` path, but use one fixed **total visual tile budget of 6 per sample**. For a 1-shot pair, that budget is split across the normal reference and target instead of allocating 12 tiles independently to each image. This follows InternVL's multi-image `n_max` design and is sized for L4 memory.

The notebook intentionally does **not** force 4-bit quantization just to fit a T4. If VRAM is below 20 GB, models are marked `skipped_insufficient_gpu`; use an L4/A100-class runtime.

## Colab smoke test

Open `notebooks/lvlm_baselines_colab.ipynb` from `main` in Google Colab and run cells from top to bottom.

Because this repository is private, add only:

- `GITHUB_TOKEN`: fine-grained/read-only token that can clone this repository

No OpenAI, Gemini, or Luxia API secret is required.

### Source ZIP upload

Do **not** mount Google Drive. In the Colab left sidebar, upload these files directly into `/content`:

- `unsupervised_anomaly_detection.zip`
- `supervised_fault_classification.zip`

Every requested model receives the identical evaluation regimes:

```text
Zero-shot 20
→ 1-shot normal reference 20
→ Structured 20
```

Before the 60-row smoke starts, each model must first pass:

```text
Zero-shot probe 1
+ 1-shot probe 1
```

This prevents a broken adapter from repeating the same failure across all 60 rows.

Outputs are written under `/content/lvlm_insplad_results/` and include:

- per-model prediction JSONL
- per-model metric CSVs
- `leaderboard.csv`
- `prediction_review.csv`
- `run_status.json`
- `run_metadata.json`
- final review ZIP

Only models with **zero row-level inference errors** are included in the scored leaderboard. Partial models are excluded.

`prediction_review.csv` now includes **all requested models**, including probe/load/row failures, so the actual exception text is preserved for the next debugging pass. `run_status.json` also stores up to five row-error examples and peak VRAM per model.

`run_metadata.json` records the benchmark commit, Transformers version, requested model IDs, GPU/VRAM, model probe results, QA counts, model registry, and source ZIP paths.

## Next evaluation step

1. Get all seven models through the 20/20/20 smoke with zero row errors.
2. Compare zero-shot vs 1-shot gains and current vs MMAD-era model generations.
3. Inspect class/category bias from `prediction_review.csv`.
4. Only after smoke stability, scale the same contract to the full 6,481 / 6,417 evaluation sets.

## Evaluation CLI

```powershell
python scripts/evaluate.py --task anomaly --qa data/qa/anomaly_zero_shot.jsonl --predictions predictions/anomaly_zero_shot.jsonl --regime zero_shot
python scripts/evaluate.py --task anomaly --qa data/qa/anomaly_one_shot.jsonl --predictions predictions/anomaly_one_shot.jsonl --regime one_shot
python scripts/evaluate.py --task explainability --qa data/qa/defect_qa.jsonl --predictions predictions/defect.jsonl
```
