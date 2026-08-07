# Source data access status

The authoritative Google Drive files are confirmed:

- `unsupervised_anomaly_detection.zip` — 1,170,967,719 bytes
- `supervised_fault_classification.zip` — 874,753,167 bytes

The connected Drive interface cannot download these >100 MB archives directly, but their **locally generated ZIP manifests were supplied and analyzed**. Therefore folder structure, asset categories, splits, image counts, directory-derived labels, annotation-file existence, Benchmark Core, QA counts, and ZIP CRC32 overlap candidates are now verified from the actual archives.

## Still requires local ZIP byte access

Two checks cannot be completed from central-directory manifests alone:

1. SHA256 confirmation of the 8,990 cross-dataset overlap candidates and JPEG dimensions.
2. The text contents/schema of the five `annotations/image_list_val.txt` files.

Run the provided `scripts/hash_overlap.py` and the `--text-preview-json` option of `scripts/inspect_zip_manifest.py` against the local ZIPs to complete those two source-level checks. No raw ZIP or image data should be committed to GitHub.
