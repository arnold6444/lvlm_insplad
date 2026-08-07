# Data issues and benchmark decisions

## Verified issues

- `supervised_fault_classification` has `train/val` only; no test split was observed. Benchmark v1 therefore uses `val` for structured explainability evaluation.
- Supervised normal labels are inconsistent: both `good` and `normal` occur. Both are normalized to benchmark `normal`, while source labels are preserved.
- Supervised raw labels are `bird-nest`, `corrosão`, `good`, `missing-cap`, `normal`, and `rust`.
- `corrosão` is mechanically normalized to `corrosao`; it is **not** semantically merged with `rust`.
- Cross-dataset strong overlap candidates: **8,990**, using filename + uncompressed size + ZIP CRC32. CRC32 is not cryptographic proof; `scripts/hash_overlap.py` confirms them with SHA256 and JPEG dimensions.
- Unsupervised has **1** cross-split duplicate-content candidate by size+CRC32. Supervised has **0** train/val duplicate-content candidates by the same proxy.
- The unsupervised duplicate candidate is:
  - `glass-insulator/test/good/Fotos 16-11-2020_DJI_0052_cadeia_isolador_vidro_1487.jpg`
  - `glass-insulator/train/good/Fotos 16-11-2020_DJI_0052_cadeia_isolador_vidro_1488.jpg`
- Five annotation-like `.txt` files exist under each supervised asset as `annotations/image_list_val.txt`; the current manifests do not contain their text, so the schema is not guessed.
- Supervised training filenames contain many numeric suffix variants such as `_0` ... `_8`, especially for defect samples, indicating possible augmentation. These are preserved rather than silently deduplicated.

## One-shot leakage policy

- Same asset category only.
- Train normal only.
- One fixed reference per asset with seed 42.
- Any train candidate whose size+CRC32 fingerprint appears in that asset's test split is excluded before selection.
- The same references are used for every LVLM.
