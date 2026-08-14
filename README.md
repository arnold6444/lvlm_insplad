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

`notebooks/lvlm_baselines_colab.ipynb` preserves one common QA/prompt/evaluation contract while routing open-source models through direct Hugging Face inference and keeping the closed-model path ready for LuxiaCloud's external-model gateway. The complete smoke-run orchestration is kept directly in the notebook as step-by-step cells so the notebook can be run top-to-bottom in Colab without a separate runner script.

### Closed/API baselines intended through LuxiaCloud

Target closed models:

- OpenAI `gpt-5.3-chat-latest`
- Google `gemini-3-flash-preview`

The Luxia API key is sent with the documented `apikey` header.

**Important routing limitation:** Luxia's public Chat Completion documentation currently lists only Luxia-owned model IDs for:

```text
POST https://bridge.luxiacloud.com/luxia/v1/chat
```

That route must therefore **not** be used as a default gateway for GPT/Gemini. The earlier benchmark assumption that arbitrary external model IDs could be sent to `/luxia/v1/chat` was disproven by live smoke probes returning `HTTP 400 invalid_parameter:modelname:dynamic_luxia_model`.

Luxia's public docs do show provider-specific bridge routes for some external services, such as OpenAI STT and Google Cloud TTS, but they do not currently publish a GPT/Gemini external text/VLM endpoint or multimodal request schema. The benchmark therefore refuses to guess an endpoint.

If Luxia provides an official external GPT/Gemini endpoint or cURL example through the dashboard/support channel, add that exact route in Colab Secrets as:

```text
LUXIA_EXTERNAL_CHAT_URL
```

Only then will the notebook run the GPT/Gemini probe and smoke calls. Without that route, the two API baselines are marked `skipped_external_route_missing` instead of incorrectly calling `/luxia/v1/chat`.

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
- `LUXIA_API_KEY`: LuxiaCloud API key
- `LUXIA_EXTERNAL_CHAT_URL`: optional; add only if Luxia supplies an official GPT/Gemini external chat/VLM endpoint

The notebook never prints secret values. Missing `LUXIA_API_KEY` skips only the API baselines. Missing `LUXIA_EXTERNAL_CHAT_URL` now produces a clear routing status instead of making a known-invalid request.

The first cell also adds `/content/lvlm_insplad/src` to the current Colab kernel's `sys.path` after editable installation. This avoids the same-session `ModuleNotFoundError: No module named 'insplad_benchmark'` that can occur when `pip install -e` is executed through a subprocess after the kernel has initialized site-packages.

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

The API phase runs independently from the GPU phase. The open-source models require an L4/A100-class runtime under the current >=20 GB VRAM policy. GPT/Gemini can run without a local GPU once Luxia provides and configures the external-model route.

Outputs are written under `/content/lvlm_insplad_results/` and include:

- per-model prediction JSONL
- per-model metric CSVs
- `leaderboard.csv`
- `prediction_review.csv`
- `run_status.json`
- `run_metadata.json`
- final review ZIP

If no model completes, `leaderboard.csv` and `prediction_review.csv` still contain stable headers and the notebook prints every skip/failure reason. The diagnostic ZIP remains downloadable.

`run_metadata.json` records the benchmark commit, requested model IDs, whether the Luxia external route was configured, API probe result, backend/provider, GPU/VRAM, QA counts, source ZIP paths, and low-VRAM policy.

> External-model API calls can incur LuxiaCloud charges. Do not start the full 6,481 / 6,417 inference until the 20-sample smoke results have been reviewed.

## Evaluation CLI

```powershell
python scripts/evaluate.py --task anomaly --qa data/qa/anomaly_zero_shot.jsonl --predictions predictions/anomaly_zero_shot.jsonl --regime zero_shot
python scripts/evaluate.py --task anomaly --qa data/qa/anomaly_one_shot.jsonl --predictions predictions/anomaly_one_shot.jsonl --regime one_shot
python scripts/evaluate.py --task explainability --qa data/qa/defect_qa.jsonl --predictions predictions/defect.jsonl
```
