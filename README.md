# InsPLAD LVLM Benchmark v1

InsPLAD-based benchmark for **General Anomaly Detection** and **Structured Fault Understanding**, inspired by MMAD-style LVLM evaluation. The benchmark compares multiple VLMs under the same QA, prompt, and scoring contract.

For the dev-selected **Q1 fixed + revised Q2 + previous Q3** gold candidate, open
[`lvlm_gold_v2_q23_candidate_colab.ipynb`](notebooks/lvlm_gold_v2_q23_candidate_colab.ipynb)
in a fresh L4/A100 Colab runtime and run all cells. It prepares missing inputs automatically,
preserves previous results, and reports Q3 **presence only**, with spatial accuracy unknown.
The notebook fetches and verifies an immutable code commit. For saved PR56 predictions,
[`CPU reaggregation`](notebooks/lvlm_gold_v2_q23_candidate_reaggregate_colab.ipynb) repairs metrics
without another GPU run. See [protocol, failure gates and recovery](docs/q23_candidate_gold.md).

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

The final baseline table uses this Aligned Core so every model is evaluated on the **same query images** for anomaly detection, normal-reference anomaly detection, and structured defect understanding.

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

The full QA and Aligned Core files are rebuilt from the two source ZIPs when needed:

- `data/qa/anomaly_zero_shot.jsonl` — 6,481
- `data/qa/anomaly_one_shot.jsonl` — 6,481
- `data/qa/defect_qa.jsonl` — 6,417
- `reports/aligned_core.csv` — 1,378 same-image links

Committed smoke samples:

- `samples/anomaly_zero_shot_20.jsonl`
- `samples/anomaly_one_shot_20.jsonl`
- `samples/defect_20.jsonl`
- `samples/aligned_core_20.jsonl`

## Colab notebooks

### Smoke / execution check

- `notebooks/lvlm_baselines_colab.ipynb` — four non-overlapping Hugging Face VLM family representatives, 20/20/20 smoke
- `notebooks/luxia_pipeline_smoke_colab.ipynb` — Luxia Vision (Document AI) → Luxia Chat two-stage API smoke

Smoke contract:

```text
Zero-shot 20
→ 1-shot normal reference 20
→ Structured 20
```

### Final HF baseline table

- `notebooks/lvlm_baselines_full_colab.ipynb` — **Aligned Core 1,378 same query images**

This notebook rebuilds the verified Aligned Core from the source ZIPs, then evaluates:

```text
Image only anomaly: 1,378
+ normal reference anomaly: 1,378
structured anomaly + defect class: 1,378
```

Predictions are checkpointed to:

```text
MyDrive/lvlm_insplad_results/hf_full_aligned_core_v1/
```

Successful rows are reused on the next run, so the four models can be run one at a time across multiple Colab sessions.

### Retrieved 1-shot+ experiment

`1-shot+` keeps the fixed one-shot prompt and evaluation contract, but replaces the asset-wide fixed normal with a query-specific reference:

```text
Aligned Core query
→ same-asset unsupervised train-normal candidates
→ exclude every train candidate whose size+ZIP-CRC32 fingerprint occurs in test
→ facebook/dinov2-base@f9e44c814b77203eaa57a6bdbbd535f21ede1415 CLS embedding
→ deterministic cosine Top-1
→ [retrieved normal reference, query] LVLM input
```

Generated artifacts are separate from the fixed baseline:

- `one_shot_plus.jsonl` — retrieved QA with unchanged query ID/image/GT
- `one_shot_plus_references.csv` — fixed/retrieved paths, cosine score, model commit, and exact cache SHA-256 provenance
- `reference_embeddings.npz` — pickle-free incremental embedding cache
- `one_shot_plus_smoke_25.jsonl` — deterministic label-balanced, asset-round-robin smoke QA
- `hf_one_shot_plus_v1/` — separate predictions and metrics

The plus experiment uses only Qwen3.5-4B, InternVL3.5-8B, and MiniCPM-V-4.5. LLaVA remains available to the legacy baseline but is intentionally rejected by `one_shot_plus` mode.

Top-1 selection is deterministic for the exact pinned embedding cache (recorded by SHA-256). A cold re-embedding on different GPU/runtime kernels can produce tiny floating-point differences near a tie, so reuse the persisted cache when comparing or resuming experiments.

Open `notebooks/lvlm_one_shot_plus_colab.ipynb` for the recommended end-to-end path. It snapshots aligned zero/fixed/structured/retrieved QA, persists the commit-pinned DINO cache in Drive, runs a 25-row smoke by default, and leaves `RUN_FULL = False`; the 1,378-row LVLM inference is never started unless that guard is changed explicitly.

For an already rebuilt benchmark/Aligned Core, the equivalent builder command is:

```powershell
python -m pip install -e ".[retrieval,hf]"
python scripts/build_one_shot_plus.py `
  --anomaly-manifest work/manifests/unsup.csv `
  --unsup-zip C:\path\to\unsupervised_anomaly_detection.zip `
  --base-one-shot-qa work/benchmark/data/qa/anomaly_one_shot.jsonl `
  --aligned-core work/benchmark/reports/aligned_core.csv `
  --out-qa work/aligned_qa/one_shot_plus.jsonl `
  --out-manifest work/retrieval/one_shot_plus_references.csv `
  --cache work/retrieval/reference_embeddings.npz `
  --smoke-out work/aligned_qa/one_shot_plus_smoke_25.jsonl `
  --smoke-size 25 `
  --model-revision f9e44c814b77203eaa57a6bdbbd535f21ede1415 `
  --device cuda
```

Run the guarded three-model smoke on an L4/A100-class runtime:

```powershell
python scripts/run_hf_aligned_core_full.py `
  --experiment one_shot_plus --smoke-size 25 `
  --qa-dir work/aligned_qa `
  --unsup-zip C:\path\to\unsupervised_anomaly_detection.zip `
  --sup-zip C:\path\to\supervised_fault_classification.zip `
  --out-dir hf_one_shot_plus_v1/smoke_25 `
  --models qwen3_5_4b internvl3_5_8b minicpm_v_4_5
```

`scripts/evaluate.py --regime one_shot_plus` writes overall, per-asset, and unweighted asset-macro anomaly metrics with TP/FP/TN/FN, Precision, Recall, and Specificity. Smoke and full runs must use different output directories; the runner rejects an incompatible or ambiguous existing result root. After complete full predictions exist, compare the three regimes and yoke-suspension FP directly (the analyzer rejects partial, wrong-model, or stale-reference plus results):

```powershell
python scripts/analyze_aligned_core_results.py `
  --qa-dir work/aligned_qa `
  --results-dir hf_full_aligned_core_v1 `
  --one-shot-plus-qa work/aligned_qa/one_shot_plus.jsonl `
  --one-shot-plus-results-dir hf_one_shot_plus_v1/full `
  --out-dir hf_one_shot_plus_v1/comparison
```

The comparison includes `anomaly_asset_macro.csv`, `fixed_vs_retrieved_overall.csv`, `fixed_vs_retrieved_by_asset.csv`, and `yoke_fp_comparison.csv`.

## 4-model Hugging Face benchmark

To avoid duplicate model families, only one representative per family is kept:

- `Qwen/Qwen3.5-4B` — Qwen family
- `OpenGVLab/InternVL3_5-8B` — InternVL family
- `openbmb/MiniCPM-V-4_5` — MiniCPM family
- `llava-hf/llava-onevision-qwen2-7b-ov-hf` — LLaVA family

Removed from the active benchmark because a newer model from the same family is already included:

- `Qwen/Qwen2.5-VL-7B-Instruct`
- `Qwen/Qwen3-VL-8B-Instruct`
- `OpenGVLab/InternVL2_5-8B`

`Qwen3.5-4B` is used instead of larger Qwen3.5 variants because the benchmark environment is an NVIDIA L4 with about 22 GB VRAM. Qwen3.5 support requires the Transformers v5 generation; the Colab notebooks pin `transformers==5.14.0` for reproducibility.

### GPU-aware adapter policy

- Qwen3.5 uses `AutoProcessor + AutoModelForMultimodalLM` with `enable_thinking=False`.
- MiniCPM-V 4.5 uses its official `AutoModel + chat()` path.
- LLaVA-OneVision uses the official `AutoProcessor + AutoModelForMultimodalLM` path.
- InternVL3.5 uses its official `chat()` path with one fixed total visual tile budget of 6 per sample.
- The notebook does not force 4-bit quantization just to fit a T4. Use an L4/A100-class runtime.

### Run the final Hugging Face baseline

Open `notebooks/lvlm_baselines_full_colab.ipynb` from `main` in Google Colab and run cells from top to bottom.

Because this repository is private, add this Colab Secret:

- `GITHUB_TOKEN`: fine-grained/read-only token that can clone this repository

The default cell requests all four models. For long Colab sessions, set `MODELS_TO_RUN` to one model at a time. The fixed Drive result directory preserves previous successful rows.

## Luxia pipeline smoke

Open `notebooks/luxia_pipeline_smoke_colab.ipynb` from `main` in Google Colab.

Add these Colab Secrets:

- `GITHUB_TOKEN`
- `LUXIA_API_KEY`

GPU is not required for the Luxia API pipeline. The notebook evaluates the same Zero-shot 20 / 1-shot 20 / Structured 20 smoke contract and exports `leaderboard.csv`, `prediction_review.csv`, `run_status.json`, and `run_metadata.json`.

The Luxia runner reuses successful predictions from the latest `luxia_smoke_*.zip`, retries failed or missing rows only, and still produces a leaderboard if a row-level API failure remains by counting that row as invalid.

## Source ZIPs from Google Drive

Put these files anywhere under Google Drive `MyDrive`:

- `unsupervised_anomaly_detection.zip`
- `supervised_fault_classification.zip`

The Colab notebooks mount Drive and find each filename automatically. If the same filename exists more than once, set `DRIVE_DATA_DIR` to the folder containing the dataset ZIPs.

## Full baseline scoring

The final Aligned Core leaderboard reports:

- Zero-shot anomaly Balanced Accuracy / F1
- Normal-reference anomaly Balanced Accuracy / F1
- Defect Balanced Accuracy / Macro-F1
- Joint Accuracy
- False Defect Rate
- Consistency Error
- Structured Invalid Output Rate

These structured metrics are an **explainability proxy / fault-understanding evaluation**. InsPLAD does not provide natural-language explanation, location, or appearance ground truth; those require a separate reviewed subset for the later method/ideation stage.

## Optional balanced-100 utility

`scripts/build_balanced_samples.py` remains available for quick intermediate experiments, but the current final baseline table goes directly from the 20-row smoke to the **Aligned Core 1,378** evaluation.

## Evaluation CLI

```powershell
python scripts/evaluate.py --task anomaly --qa data/qa/anomaly_zero_shot.jsonl --predictions predictions/anomaly_zero_shot.jsonl --regime zero_shot
python scripts/evaluate.py --task anomaly --qa data/qa/anomaly_one_shot.jsonl --predictions predictions/anomaly_one_shot.jsonl --regime one_shot
python scripts/evaluate.py --task explainability --qa data/qa/defect_qa.jsonl --predictions predictions/defect.jsonl
```
