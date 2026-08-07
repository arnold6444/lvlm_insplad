# QA sample review

Deterministic stratified review samples use `seed=42`.

## Track A — zero-shot / one-shot

- `samples/anomaly_zero_shot_20.jsonl`: 20 examples covering all 17 Track A assets.
- `samples/anomaly_one_shot_20.jsonl`: the exact same 20 test images/GT plus fixed same-asset train-normal references.
- Normal-only assets are represented as normal examples; mixed-class assets include anomalous examples where selected by the deterministic stratifier.

## Track B — structured fault understanding

- `samples/defect_20.jsonl`: 20 examples across all 5 supervised assets.
- Each row records asset-specific `valid_defect_classes` and `{anomaly, defect_class}` GT.

## Aligned Core

- `samples/aligned_core_20.jsonl`: 20 exact same-image links across all 5 shared assets.
- Each row links the Track A anomaly ID, Track B structured ID, one-shot reference image, anomaly GT, and defect GT.
- The 6 exact-image binary-GT conflicts are excluded from this clean sample pool.

All full QA files passed automated validation before these review samples were written.
