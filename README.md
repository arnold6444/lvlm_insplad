# InsPLAD LVLM Benchmark v1

InsPLAD-based benchmark for **General Anomaly Detection** and **Structured Fault Understanding**, inspired by MMAD-style LVLM evaluation. The benchmark compares multiple VLMs under the same QA, prompt, and scoring contract.

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

## Baseline policy — Hugging Face VLMs only

`notebooks/lvlm_baselines_colab.ipynb` is the single Colab execution entrypoint. It keeps one QA/prompt/evaluation contract and runs every Hugging Face VLM sequentially on the same smoke set.

OpenAI/Gemini API baselines have been removed.

### Why Luxia is not in the image benchmark yet

LuxiaCloud publicly documents:

- text chat: `POST https://bridge.luxiacloud.com/luxia/v1/chat`
- vision: `POST https://bridge.luxiacloud.com/luxia/v1/document-ai`

The public Vision documentation explicitly states that Document AI is specialized for **document images** and is **not intended for general image processing**. InsPLAD contains power-line asset photographs, so using Document AI as a competing anomaly-detection VLM would not be a fair or meaningful comparison.

Therefore Luxia is **deferred, not silently substituted**. If Luxia later exposes a general-image VLM endpoint/model suitable for arbitrary photographs, it can be added as a separate baseline under the same QA contract.

### Open-source/GPU baselines

Current models:

- `Qwen/Qwen3-VL-8B-Instruct`
- `Qwen/Qwen2.5-VL-7B-Instruct`
- `OpenGVLab/InternVL3_5-8B`
- `OpenGVLab/InternVL2_5-8B`
- `openbmb/MiniCPM-V-4_5`
- `llava-hf/llava-onevision-qwen2-7b-ov-hf`

Why these models:

- MMAD's official evaluation code directly supports Qwen, InternVL, MiniCPM, and LLaVA families.
- MMAD later added a dedicated `qwen2_5_vl_query.py`, so Qwen2.5-VL is included as a direct MMAD-era comparison anchor.
- Qwen3-VL, InternVL3.5, and MiniCPM-V-4.5 keep a current-generation view of the same major open-source families.
- InternVL2.5 provides an additional MMAD-era anchor while reusing the same InternVL chat adapter.

Qwen3-VL, Qwen2.5-VL, and LLaVA use the generic Transformers image-text pipeline. InternVL3.5/2.5 and MiniCPM-V use model-specific `AutoModel + chat()` adapters.

The notebook intentionally does **not** force 4-bit quantization just to fit a T4. If VRAM is below 20 GB, models are marked `skipped_insufficient_gpu`; use an L4/A100-class runtime for the paper baseline.

## Colab smoke test

Open `notebooks/lvlm_baselines_colab.ipynb` from `main` in Google Colab and run cells from top to bottom.

Because this repository is private, add only:

- `GITHUB_TOKEN`: fine-grained/read-only token that can clone this repository

No OpenAI, Gemini, or Luxia API secret is required for the current benchmark.

### Source ZIP upload

Do **not** mount Google Drive. In the Colab left sidebar, upload these files directly into `/content`:

- `unsupervised_anomaly_detection.zip`
- `supervised_fault_classification.zip`

Every requested model receives the identical smoke set:

```text
Zero-shot 20
→ One-shot 20
→ Structured 20
```

Outputs are written under `/content/lvlm_insplad_results/` and include:

- per-model prediction JSONL
- per-model metric CSVs
- `leaderboard.csv`
- `prediction_review.csv`
- `run_status.json`
- `run_metadata.json`
- final review ZIP

Only models with **zero row-level inference errors** are included in the scored leaderboard. A model that partially fails is marked `completed_with_row_errors` and excluded from scoring instead of being treated as a valid completed run.

If no model completes, `leaderboard.csv` and `prediction_review.csv` still contain stable headers and the notebook prints every skip/failure reason.

`run_metadata.json` records the benchmark commit, requested HF model IDs, GPU/VRAM, QA counts, model registry, source ZIP paths, and the fact that Luxia general-vision evaluation is deferred because the public API is document-only.

## Evaluation CLI

```powershell
python scripts/evaluate.py --task anomaly --qa data/qa/anomaly_zero_shot.jsonl --predictions predictions/anomaly_zero_shot.jsonl --regime zero_shot
python scripts/evaluate.py --task anomaly --qa data/qa/anomaly_one_shot.jsonl --predictions predictions/anomaly_one_shot.jsonl --regime one_shot
python scripts/evaluate.py --task explainability --qa data/qa/defect_qa.jsonl --predictions predictions/defect.jsonl
```
