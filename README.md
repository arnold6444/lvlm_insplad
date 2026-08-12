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

## Baseline policy — Luxia API for closed models, GPU for open-source models

`notebooks/lvlm_baselines_colab.ipynb` preserves one common QA/prompt/evaluation contract while routing closed models through LuxiaCloud and open-source models through direct Hugging Face inference. The complete smoke-run orchestration is kept directly in the notebook as step-by-step cells so the notebook can be run top-to-bottom in Colab without a separate runner script.

### Closed/API baselines through LuxiaCloud

These use one LuxiaCloud API key and do **not** need a GPU:

- OpenAI `gpt-5.3-chat-latest`
- Google `gemini-3-flash-preview`

The benchmark sends external models to LuxiaCloud's chat bridge with the `apikey` header. The default endpoint is:

```text
https://bridge.luxiacloud.com/luxia/v1/chat
```

If Saltlux provides a dedicated external-model route later, set the optional environment variable `LUXIA_EXTERNAL_CHAT_URL` without changing benchmark code.

Because the public Luxia chat documentation does not currently spell out the external-model multimodal request schema, the notebook performs a **single-image probe for each external model before the smoke benchmark**. If the bridge rejects the model ID or multimodal content, that model is marked `skipped_api_probe_failed` and the remaining 60 calls are not attempted.

### Open-source/GPU baselines

These run directly with Hugging Face on a rented GPU, one model at a time:

- `Qwen/Qwen3-VL-8B-Instruct`
- `OpenGVLab/InternVL3_5-8B`
- `openbmb/MiniCPM-V-4_5`
- `llava-hf/llava-onevision-qwen2-7b-ov-hf`

Qwen3-VL and LLaVA use the generic Transformers image-text pipeline. InternVL3.5 and MiniCPM-V use their model-specific `AutoModel + chat()` adapters because their remote-code configs are not reliably compatible with `AutoModelForImageTextToText`.

The notebook intentionally does **not** switch these models to 4-bit quantization just because a T4 is available. If VRAM is below 20 GB, direct HF baselines are marked `skipped_insufficient_gpu`; use an L4/A100-class runtime for the paper baseline instead.

## Colab smoke test

Open `notebooks/lvlm_baselines_colab.ipynb` from `main` in Google Colab and run the cells from top to bottom.

Because this repository is private, add the following values in **Colab → Secrets** before running:

- `GITHUB_TOKEN`: fine-grained/read-only token that can clone this repository
- `LUXIA_API_KEY`: LuxiaCloud key used for both GPT and Gemini external-model baselines

The notebook never prints the secret value. Missing `LUXIA_API_KEY` skips only the API baselines.

### Source ZIP upload

Do **not** mount Google Drive. In the Colab left sidebar, open **Files** and upload these two files directly into the runtime root `/content`:

- `unsupervised_anomaly_detection.zip`
- `supervised_fault_classification.zip`

The notebook checks `/content` for those exact filenames and fails early with a clear message if either file is missing.

Every requested model receives the identical smoke set:

```text
Zero-shot 20
→ One-shot 20
→ Structured 20
```

The API phase runs independently from the GPU phase. GPT/Gemini can therefore be evaluated without a suitable local GPU; the open-source models can then run on an L4/A100-class runtime under the same benchmark commit and QA contract.

Outputs are written under `/content/lvlm_insplad_results/` and include:

- per-model prediction JSONL
- per-model metric CSVs
- `leaderboard.csv`
- `prediction_review.csv`
- `run_status.json`
- `run_metadata.json`
- final review ZIP

`run_metadata.json` records the benchmark commit, requested model IDs, Luxia endpoint, API probe result, backend/provider, GPU/VRAM, QA counts, source ZIP paths, and low-VRAM policy.

> External-model API calls can incur LuxiaCloud charges. Do not start the full 6,481 / 6,417 inference until the 20-sample smoke results have been reviewed.

## Evaluation CLI

```powershell
python scripts/evaluate.py --task anomaly --qa data/qa/anomaly_zero_shot.jsonl --predictions predictions/anomaly_zero_shot.jsonl --regime zero_shot
python scripts/evaluate.py --task anomaly --qa data/qa/anomaly_one_shot.jsonl --predictions predictions/anomaly_one_shot.jsonl --regime one_shot
python scripts/evaluate.py --task explainability --qa data/qa/defect_qa.jsonl --predictions predictions/defect.jsonl
```
