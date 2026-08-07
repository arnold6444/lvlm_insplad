# InsPLAD LVLM Benchmark v1

InsPLAD-based benchmark for **General Anomaly Detection** and **Structured Fault Understanding**, inspired by MMAD-style LVLM evaluation. The benchmark compares multiple LVLMs under the same QA, prompt, and scoring contract.

## Benchmark design

### Track A — General Anomaly Detection
Source: `unsupervised_anomaly_detection.zip`

- 17 asset categories
- official `test`: **6,481** images
- Zero-shot: target test image only
- One-shot: same target test image + one fixed same-asset normal image from `train`
- 14 assets contain both normal and anomalous test GT
- 3 assets are normal-only in test: `damper-preformed`, `spacer`, `yoke`

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

## Google Colab — multi-model LVLM baseline smoke test

Open `notebooks/lvlm_baselines_colab.ipynb` in Google Colab and select a rented GPU runtime, preferably **24 GB+ VRAM** for the 7B–8B models.

The initial open-source baseline registry contains:

- `Qwen/Qwen3-VL-8B-Instruct`
- `OpenGVLab/InternVL3_5-8B`
- `openbmb/MiniCPM-V-4_5`
- `llava-hf/llava-onevision-qwen2-7b-ov-hf`

The notebook intentionally runs **one model at a time**:

```text
model A load
  → Zero-shot 20
  → One-shot 20
  → Structured 20
  → predictions + metrics 저장
  → GPU unload
model B load
  → same 60 QA
  → ...
```

This keeps the comparison fair: every LVLM sees the same sampled images, same reference images, same prompts, same deterministic decoding policy, and the same `evaluate.py` metrics.

Outputs are saved to Google Drive under `MyDrive/lvlm_insplad_results/` and include:

- per-model prediction JSONL
- per-model metric CSVs
- `leaderboard.csv`
- `prediction_review.csv`
- `run_status.json`
- `run_metadata.json`
- final upload-ready result ZIP

`MODELS_TO_RUN` in the notebook can be edited to run only part of the registry first. More Hugging Face LVLMs can be added by extending `MODEL_REGISTRY`. Proprietary GPT/Gemini API baselines can be added later using the same QA/output contract.

Do **not** start full 6,481 / 6,417 inference until the 20-sample multi-model smoke test is reviewed.

## Evaluation CLI

```powershell
python scripts/evaluate.py --task anomaly --qa data/qa/anomaly_zero_shot.jsonl --predictions predictions/anomaly_zero_shot.jsonl --regime zero_shot
python scripts/evaluate.py --task anomaly --qa data/qa/anomaly_one_shot.jsonl --predictions predictions/anomaly_one_shot.jsonl --regime one_shot
python scripts/evaluate.py --task explainability --qa data/qa/defect_qa.jsonl --predictions predictions/defect.jsonl
```
