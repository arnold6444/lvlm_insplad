# Source data verification status

The authoritative source ZIPs are:

- `unsupervised_anomaly_detection.zip` — 1,170,967,719 bytes
- `supervised_fault_classification.zip` — 874,753,167 bytes

The connected Drive interface cannot directly download these >100 MB archives, so source-level inspection was completed with small metadata files generated locally from the exact ZIPs.

## Completed checks

- ZIP folder/member manifests supplied and parsed.
- Asset categories, splits, image counts, and directory-derived labels verified.
- Five supervised `annotations/image_list_val.txt` files detected.
- Annotation text previews supplied; line schema verified as `<relative_image_path> <integer_class_id>`.
- Annotation integer IDs confirmed to be asset-local class indices rather than binary anomaly labels.
- 8,990 manifest overlap candidates hashed locally.
- SHA256 matches: **8,990 / 8,990**.
- SHA256 mismatches: **0**.
- JPEG dimensions match: **8,990 / 8,990**.
- Exact unsupervised-test ↔ supervised-val overlap: **1,384**.
- Binary GT conflicts among those exact images: **6**.
- Clean Aligned Core: **1,378**.

No raw ZIP or image data is committed to GitHub.
