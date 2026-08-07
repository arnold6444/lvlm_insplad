# InsPLAD LVLM Benchmark v1

Tooling for an InsPLAD-based benchmark of:

1. anomaly detection (zero-shot / one-shot), and
2. structured explainability (`anomaly` + `defect_class`).

The repository intentionally does **not** hard-code InsPLAD asset categories, splits, or defect names. Those values must be derived from the authoritative fault-dataset archives after their actual contents are inspected.

## Setup

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Inspect large ZIPs without extracting them

```bash
python scripts/inspect_zip_manifest.py "D:/datasets/unsupervised_anomaly_detection.zip" --out-csv reports/unsupervised_zip_manifest.csv --out-json reports/unsupervised_zip_summary.json
python scripts/inspect_zip_manifest.py "D:/datasets/supervised_fault_classification.zip" --out-csv reports/supervised_zip_manifest.csv --out-json reports/supervised_zip_summary.json
```

## Evaluate model outputs

Predictions are JSONL rows with `id` and raw model `output`.

```bash
python scripts/evaluate.py --task anomaly --qa data/qa/anomaly_zero_shot.jsonl --predictions predictions/anomaly_zero_shot.jsonl --regime zero_shot
python scripts/evaluate.py --task explainability --qa data/qa/defect_qa.jsonl --predictions predictions/defect.jsonl
```

## Validate QA

```bash
python scripts/validate_qa.py data/qa/defect_qa.jsonl --assets configs/assets.json --defects configs/defects.json --image-index reports/image_index.csv
```

See `reports/data_access_blocked.md` for the current source-data access boundary.
