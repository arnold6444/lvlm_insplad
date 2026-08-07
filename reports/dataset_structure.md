# InsPLAD fault dataset structure (verified)

## Track A — `unsupervised_anomaly_detection.zip`

- Asset categories: **17**
- Train images: **21,139**
- Test images: **6,481**
- Splits: `train`, `test`
- Train contains normal (`good`) reference candidates.
- Track A evaluates all 17 assets on official `test`.

Test defect folders observed across assets include `rust`, `missingcap`, `nest`, `peeling-paint`, and `torned-up`; three assets (`damper-preformed`, `spacer`, `yoke`) have normal-only test data.

## Track B — `supervised_fault_classification.zip`

- Asset categories: **5**
- Train images: **5,108**
- Val images: **6,417**
- Splits: `train`, `val`; no test split observed.
- Structured evaluation uses official `val`.
- Five `annotations/image_list_val.txt` files are present.

### Shared assets

- `glass-insulator`
- `lightning-rod-suspension`
- `polymer-insulator-upper-shackle`
- `vari-grip`
- `yoke-suspension`

### Supervised defect folders

- `glass-insulator`: `good`, `missing-cap`
- `lightning-rod-suspension`: `good`, `rust`
- `polymer-insulator-upper-shackle`: train has `normal`, `corrosão`; val has `good`, `rust`
- `vari-grip`: `good`, `bird-nest`, `rust`
- `yoke-suspension`: `good`, `rust`

## Annotation schema

Verified line schema: `<relative_image_path> <integer_class_id>`.

The integer is an asset-local class ID, not a universal binary anomaly label. `vari-grip` proves this: `bird-nest→0`, `good→1`, `rust→2`. Semantic benchmark GT therefore comes from folder defect names.

## Exact overlap and Aligned Core

- Exact SHA256-identical images across ZIPs: **8,990 / 8,990 candidates**.
- Exact `unsupervised test ↔ supervised val` pairs: **1,384**.
- Binary GT conflicts among those exact pairs: **6**.
- Clean Aligned Core: **1,378**.

Aligned Core definition: `unsupervised test ∩ supervised val ∩ exact SHA256 match ∩ binary GT agreement`.
