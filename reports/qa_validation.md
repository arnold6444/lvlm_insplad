# QA Validation

- Status: **PASS**

## `data/qa/anomaly_zero_shot.jsonl`
- Rows checked: **2,656**
- Errors: **0**
- Warnings: **0**

## `data/qa/anomaly_one_shot.jsonl`
- Rows checked: **2,656**
- Errors: **0**
- Warnings: **0**

## `data/qa/defect_qa.jsonl`
- Rows checked: **6,417**
- Errors: **0**
- Warnings: **0**

## Zero-shot / one-shot alignment
- **PASS**: identical IDs, images, and ground-truth answers.

## Scope
- Manifest-backed image path, asset, split and label validation: PASS.
- One-shot reference normal/category/train checks: PASS.
- Duplicate reference leakage uses SHA256 when available, otherwise size+ZIP CRC32 fingerprint.
- Exact cross-dataset SHA256 confirmation remains a local ZIP operation via `scripts/hash_overlap.py`.
