# InsPLAD LVLM Benchmark v1

InsPLAD-based baseline benchmark for **Anomaly Detection** (zero-shot / one-shot) and **Structured Explainability** (`anomaly` + `defect_class`). Benchmark v1 excludes LVLM training, large-scale inference, free-form descriptions, localization, and LVLM Judge.

## Verified source structure

Derived from manifests of the two authoritative Google Drive ZIPs:

- `unsupervised_anomaly_detection.zip`: 17 assets, `train/test`, 27,620 JPGs.
- `supervised_fault_classification.zip`: 5 assets, `train/val`, 11,525 JPGs + five `annotations/image_list_val.txt` files.
- Benchmark Core: the five shared assets.
- Core unsupervised test QA: **2,656** images.
- Structured QA: supervised has no test split, so official **val** is used: **6,417** images.

## Setup

Run from the repository root (not `C:\Windows\System32`).

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Build manifests without extracting ZIPs

```powershell
python scripts/inspect_zip_manifest.py "D:/datasets/unsupervised_anomaly_detection.zip" --out-csv reports/unsupervised_zip_manifest.csv --out-json reports/unsupervised_zip_summary.json
python scripts/inspect_zip_manifest.py "D:/datasets/supervised_fault_classification.zip" --out-csv reports/supervised_zip_manifest.csv --out-json reports/supervised_zip_summary.json --text-preview-json reports/supervised_annotation_previews.json
```

## Build QA + reports

```powershell
python scripts/build_benchmark.py `
  --anomaly-manifest reports/unsupervised_zip_manifest.csv `
  --fault-manifest reports/supervised_zip_manifest.csv
```

This creates `data/qa/anomaly_zero_shot.jsonl`, `data/qa/anomaly_one_shot.jsonl`, `data/qa/defect_qa.jsonl`, mappings, reports, and 20-sample review files.

## Validate QA

```powershell
python scripts/validate_qa.py `
  --anomaly-manifest reports/unsupervised_zip_manifest.csv `
  --fault-manifest reports/supervised_zip_manifest.csv
```

## Confirm exact image overlap with SHA256

Manifest comparison finds 8,990 strong overlap candidates via filename + uncompressed size + ZIP CRC32. Confirm exact overlap and JPEG dimensions without extracting:

```powershell
python scripts/hash_overlap.py `
  --anomaly-zip "D:/datasets/unsupervised_anomaly_detection.zip" `
  --fault-zip "D:/datasets/supervised_fault_classification.zip" `
  --anomaly-manifest reports/unsupervised_zip_manifest.csv `
  --fault-manifest reports/supervised_zip_manifest.csv `
  --out reports/image_overlap.csv
```

## Evaluate model outputs later

```powershell
python scripts/evaluate.py --task anomaly --qa data/qa/anomaly_zero_shot.jsonl --predictions predictions/anomaly_zero_shot.jsonl --regime zero_shot
python scripts/evaluate.py --task anomaly --qa data/qa/anomaly_one_shot.jsonl --predictions predictions/anomaly_one_shot.jsonl --regime one_shot
python scripts/evaluate.py --task explainability --qa data/qa/defect_qa.jsonl --predictions predictions/defect.jsonl
```

No LVLM inference is executed in this stage.
