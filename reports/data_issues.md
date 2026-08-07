# Data issues and benchmark decisions

## Benchmark tracks

- Track A uses all **17** unsupervised assets and official test (**6,481** images).
- Track A assets with both normal/anomalous test GT: **14**.
- Track A normal-only test assets: **3** — `damper-preformed`, `spacer`, `yoke`.
- Track B uses all **5** supervised assets and official val (**6,417** images) because no test split exists.
- Shared asset categories: **5**.

## Exact cross-dataset overlap

- Manifest overlap candidates: **8,990**.
- SHA256 confirmed matches: **8,990**.
- SHA256 mismatches: **0**.
- `unsupervised test ↔ supervised val`: **1,384** exact SHA256 pairs.
- Binary GT conflicts in that aligned split: **6**.
- Clean Aligned Core: **1,378** images.

The Aligned Core excludes exact same images whose binary normal/anomalous GT disagrees between the two dataset variants.

## Label policy

- Source normal labels `good` and `normal` are normalized to benchmark `normal`.
- `corrosão` is mechanically normalized to `corrosao` and is NOT merged with `rust`.
- `image_list_val.txt` line schema is `<relative_image_path> <integer_class_id>`.
- Annotation integer IDs are asset-local class IDs, not binary anomaly values.
- Example proving this: `vari-grip` preview maps `bird-nest→0`, `good→1`, `rust→2`.
- Therefore benchmark semantic GT is derived from defect folder names, not the numeric annotation ID.

## Leakage checks

- Unsupervised cross-split duplicate candidates: **1**.
- Supervised cross-split duplicate candidates: **0**.
- One-shot reference images come only from same-asset unsupervised train normals, seed 42.
- A train reference is excluded when its size+ZIP CRC32 content fingerprint occurs in unsupervised test.

## Aligned binary GT conflicts

- `Fotos 03-12-2020_DJI_0401_cadeia_isolador_vidro_790.jpg` — `glass-insulator`: unsupervised `good`→normal vs supervised `missing-cap`→anomalous
- `Fotos 03-12-2020_DJI_0557_cadeia_isolador_vidro_698.jpg` — `glass-insulator`: unsupervised `good`→normal vs supervised `missing-cap`→anomalous
- `02-06-2021_DJI_0073_198.jpg` — `polymer-insulator-upper-shackle`: unsupervised `good`→normal vs supervised `rust`→anomalous
- `Fotos 03-12-2020_DJI_0023_vari_grip_133.jpg` — `vari-grip`: unsupervised `nest`→anomalous vs supervised `good`→normal
- `Fotos 17-11-2020_DJI_0250_vari_grip_482.jpg` — `vari-grip`: unsupervised `nest`→anomalous vs supervised `good`→normal
- `Fotos 26-11-2020_DJI_0357_vari_grip_876.jpg` — `vari-grip`: unsupervised `nest`→anomalous vs supervised `good`→normal
