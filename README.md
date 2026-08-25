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

## Canonical Colab notebooks

Only two notebooks are used for the current smoke benchmark:

- `notebooks/lvlm_baselines_colab.ipynb` — seven Hugging Face VLM baselines
- `notebooks/luxia_pipeline_smoke_colab.ipynb` — Luxia Vision (Document AI) → Luxia Chat two-stage baseline

Both use the same 20/20/20 benchmark contract:

```text
Zero-shot 20
→ 1-shot normal reference 20
→ Structured 20
```

## 7-model Hugging Face benchmark

Models are grouped so the final analysis can compare MMAD-era anchors vs current/latest open VLMs:

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

- Qwen2.5/Qwen3 use the generic Transformers image-text pipeline.
- Qwen3.5 uses `AutoProcessor + AutoModelForMultimodalLM` with `enable_thinking=False`.
- MiniCPM-V 4.5 uses its official `AutoModel + chat()` path.
- LLaVA-OneVision uses the official `AutoProcessor + AutoModelForMultimodalLM` path.
- InternVL2.5/3.5 use their official `chat()` path with one fixed total visual tile budget of 6 per sample.
- The notebook does not force 4-bit quantization just to fit a T4. If VRAM is below 20 GB, models are marked `skipped_insufficient_gpu`; use an L4/A100-class runtime.

### Run the Hugging Face smoke

Open `notebooks/lvlm_baselines_colab.ipynb` from `main` in Google Colab and run cells from top to bottom.

Because this repository is private, add this Colab Secret:

- `GITHUB_TOKEN`: fine-grained/read-only token that can clone this repository

Use an NVIDIA L4-class or larger GPU.

## Luxia pipeline smoke

Open `notebooks/luxia_pipeline_smoke_colab.ipynb` from `main` in Google Colab.

Add these Colab Secrets:

- `GITHUB_TOKEN`
- `LUXIA_API_KEY`

GPU is not required for the Luxia API pipeline. The notebook evaluates the same Zero-shot 20 / 1-shot 20 / Structured 20 contract and exports `leaderboard.csv`, `prediction_review.csv`, `run_status.json`, and `run_metadata.json`.

## Source ZIPs from Google Drive

Put these files anywhere under Google Drive `MyDrive`:

- `unsupervised_anomaly_detection.zip`
- `supervised_fault_classification.zip`

Both canonical notebooks mount Drive and find each filename automatically. If the same filename exists more than once, set `DRIVE_DATA_DIR` to the folder containing the dataset ZIPs.

Result ZIPs are copied to:

```text
MyDrive/lvlm_insplad_results/
```

The notebooks also attempt the normal browser download after saving the result to Drive.

## Smoke-test scoring policy

Before the 60-row Hugging Face smoke starts, each HF model first passes:

```text
Zero-shot probe 1
+ 1-shot probe 1
```

Only models with zero row-level inference errors are included in the scored leaderboard. `prediction_review.csv` preserves failure details and `run_status.json` records error examples plus peak VRAM.

## Balanced 100 stage

After the smoke results are reviewed, build a deterministic intermediate comparison set before spending GPU/API time on the full benchmark:

```powershell
python scripts/build_balanced_samples.py --count 100 --seed 42
```

This writes `samples/balanced100/` with:

- anomaly zero-shot: **50 normal / 50 anomalous**, spread round-robin across asset categories
- anomaly one-shot: the **same 100 target/query IDs** as zero-shot, with their normal references
- structured: 100 rows distributed across assets first, then defect classes within each asset
- `sample_summary.json` so the exact sampling composition is auditable

The 20-row smoke remains the execution-stability gate. The balanced100 subset is the next model-comparison stage; full 6,481 / 6,417 evaluation comes only after that comparison is stable.

## Evaluation CLI

```powershell
python scripts/evaluate.py --task anomaly --qa data/qa/anomaly_zero_shot.jsonl --predictions predictions/anomaly_zero_shot.jsonl --regime zero_shot
python scripts/evaluate.py --task anomaly --qa data/qa/anomaly_one_shot.jsonl --predictions predictions/anomaly_one_shot.jsonl --regime one_shot
python scripts/evaluate.py --task explainability --qa data/qa/defect_qa.jsonl --predictions predictions/defect.jsonl
```
