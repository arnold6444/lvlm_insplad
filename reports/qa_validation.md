# QA Validation

- Status: **PASS**

## `data/qa/anomaly_zero_shot.jsonl`

- Rows checked: **6,481**
- Errors: **0**
- Warnings: **0**

## `data/qa/anomaly_one_shot.jsonl`

- Rows checked: **6,481**
- Errors: **0**
- Warnings: **0**

## `data/qa/defect_qa.jsonl`

- Rows checked: **6,417**
- Errors: **0**
- Warnings: **0**

## Zero-shot / one-shot alignment

- **PASS**: identical IDs, test images, and anomaly GT answers.

Validation includes manifest-backed image/category/split/label checks, reference normal/category/split checks, duplicate-content leakage checks, and structured GT consistency.
