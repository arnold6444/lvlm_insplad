# Data access status

The authoritative Google Drive files were resolved through `insplad_google_drive` on repository `main`:

- `unsupervised_anomaly_detection.zip` — 1,170,967,719 bytes
- `supervised_fault_classification.zip` — 874,753,167 bytes

The connected Google Drive raw-download action currently rejects files larger than 100 MB. Because neither archive can be opened in this environment, their internal folder structure, category names, split names, labels, annotation files, class counts, image overlap, and hashes are **not yet verified**.

Per the benchmark specification, no dataset-specific category/split/defect assumptions are encoded in the implementation.

## Local manifest extraction

Run these commands against the exact downloaded ZIP files; they read the ZIP central directory and do not extract images:

```bash
python scripts/inspect_zip_manifest.py "D:/datasets/unsupervised_anomaly_detection.zip" --out-csv reports/unsupervised_zip_manifest.csv --out-json reports/unsupervised_zip_summary.json
python scripts/inspect_zip_manifest.py "D:/datasets/supervised_fault_classification.zip" --out-csv reports/supervised_zip_manifest.csv --out-json reports/supervised_zip_summary.json
```

Once these small manifest files are available in the repository or chat, the dataset adapters and QA generation can be completed against the real structure without uploading the 2 GB of image data.
