# InsPLAD LVLM Benchmark v1

InsPLAD-based baseline benchmark for **Anomaly Detection** and **Structured Fault Understanding**, inspired by MMAD-style QA evaluation. v1 builds the benchmark, validation, and evaluation tooling; it does **not** fine-tune an LVLM or run large-scale inference.

## Verified benchmark design

### Track A — General Anomaly Detection

Source: `unsupervised_anomaly_detection.zip`

- 17 asset categories
- `train/test`
- 27,620 JPGs total
- official **test**: **6,481** images
- zero-shot: test image only
- one-shot: the same 6,481 test images + one fixed same-asset normal image from train (`seed=42`)
- 14 assets contain both normal and anomalous test GT
- 3 assets are normal-only in test: `damper-preformed`, `spacer`, `yoke`

### Track B — Structured Fault Understanding

Source: `supervised_fault_classification.zip`

- 5 asset categories
- `train/val` (no test split)
- 11,525 JPGs
- official **val**: **6,417** images
- output: `{"anomaly": "...", "defect_class": "..."}`
- Joint Accuracy is computed inside this supervised track only.

### Aligned Core — same-image cross-track analysis

The two ZIPs share **8,990 exact SHA256-identical images**. Of these, **1,384** are simultaneously:

- unsupervised `test`, and
- supervised `val`.

Six exact same images have conflicting binary GT between dataset variants, so the clean Aligned Core is:

**1,378 images = unsupervised test ∩ supervised val ∩ exact SHA256 match ∩ binary GT agreement**

`reports/aligned_core.csv` links the Track A anomaly ID, Track B structured ID, reference image, and GT for each clean same-image row.

## Verified supervised annotation schema

Each `annotations/image_list_val.txt` line is:

```text
<relative_image_path> <integer_class_id>
```

The integer is an **asset-local classification class ID**, not a universal normal/anomalous flag. For example, `vari-grip` uses `bird-nest→0`, `good→1`, `rust→2`. Benchmark semantic GT therefore comes from the defect folder name.

## Setup

Run from the repository root, not `C:\Windows\System32`.

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

To see every automatic test name:

```powershell
python -m pytest -v
```

## Build source manifests without extracting ZIPs

```powershell
python scripts/inspect_zip_manifest.py `
  "C:\path\unsupervised_anomaly_detection.zip" `
  --out-csv reports/unsupervised_zip_manifest.csv `
  --out-json reports/unsupervised_zip_summary.json

python scripts/inspect_zip_manifest.py `
  "C:\path\supervised_fault_classification.zip" `
  --out-csv reports/supervised_zip_manifest.csv `
  --out-json reports/supervised_zip_summary.json `
  --text-preview-json reports/supervised_annotation_previews.json
```

## Confirm exact overlap

```powershell
python scripts/hash_overlap.py `
  --anomaly-zip "C:\path\unsupervised_anomaly_detection.zip" `
  --fault-zip "C:\path\supervised_fault_classification.zip" `
  --anomaly-manifest reports/unsupervised_zip_manifest.csv `
  --fault-manifest reports/supervised_zip_manifest.csv `
  --out reports/image_overlap.csv `
  --summary reports/image_overlap_summary.json
```

## Build QA, reports, and Aligned Core

```powershell
python scripts/build_benchmark.py `
  --anomaly-manifest reports/unsupervised_zip_manifest.csv `
  --fault-manifest reports/supervised_zip_manifest.csv `
  --overlap-summary reports/image_overlap_summary.json `
  --annotation-previews reports/supervised_annotation_previews.json
```

Main outputs:

- `data/qa/anomaly_zero_shot.jsonl` — 6,481
- `data/qa/anomaly_one_shot.jsonl` — 6,481
- `data/qa/defect_qa.jsonl` — 6,417
- `reports/aligned_core.csv` — 1,378 same-image links
- `samples/*_20.jsonl` — deterministic smoke-test samples

## Validate QA

```powershell
python scripts/validate_qa.py `
  --anomaly-manifest reports/unsupervised_zip_manifest.csv `
  --fault-manifest reports/supervised_zip_manifest.csv
```

## Google Colab Qwen3-VL 20-sample baseline

Open `notebooks/qwen3_vl_baseline.ipynb` in Google Colab and select a GPU runtime. The notebook is designed to run top-to-bottom and will:

1. clone this branch and run the benchmark unit tests,
2. mount Google Drive,
3. find `unsupervised_anomaly_detection.zip` and `supervised_fault_classification.zip`,
4. copy the ZIPs to `/content` for faster repeated reads,
5. select `Qwen/Qwen3-VL-8B-Instruct` when VRAM is at least 22 GB, otherwise `Qwen/Qwen3-VL-4B-Instruct`,
6. run Zero-shot 20 / One-shot 20 / Structured 20 with deterministic greedy decoding,
7. save prediction JSONL files and invoke the existing `scripts/evaluate.py`,
8. write `smoke_review.csv`, metrics CSVs, run metadata, and a final upload-ready ZIP under `MyDrive/lvlm_insplad_results/`.

The notebook reads only the images needed for the smoke test from the source ZIPs; it does not extract the full datasets. If an 8B run runs out of GPU memory, set `MODEL_ID_OVERRIDE = 'Qwen/Qwen3-VL-4B-Instruct'` and rerun from the model-selection/load cells.

Do **not** start full 6,481 / 6,417 inference until the 20-sample outputs and invalid-output behavior have been reviewed.

## Evaluate LVLM outputs later

```powershell
python scripts/evaluate.py --task anomaly --qa data/qa/anomaly_zero_shot.jsonl --predictions predictions/anomaly_zero_shot.jsonl --regime zero_shot
python scripts/evaluate.py --task anomaly --qa data/qa/anomaly_one_shot.jsonl --predictions predictions/anomaly_one_shot.jsonl --regime one_shot
python scripts/evaluate.py --task explainability --qa data/qa/defect_qa.jsonl --predictions predictions/defect.jsonl
```

The next stage is to **execute the Colab 20-sample smoke test and inspect its saved result ZIP**, not full-dataset inference.
