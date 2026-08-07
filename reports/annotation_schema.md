# Supervised annotation schema

- Line schema: `<relative_image_path> <integer_class_id>`
- The trailing integer is an asset-local classification class ID, not a binary anomaly label.
- Benchmark GT is derived from the defect folder name.

## Observed class IDs from local previews

- `glass-insulator`: `good` → [0], `missing-cap` → [1]
- `lightning-rod-suspension`: `good` → [0], `rust` → [1]
- `polymer-insulator-upper-shackle`: `good` → [0], `rust` → [1]
- `vari-grip`: `bird-nest` → [0], `good` → [1], `rust` → [2]
- `yoke-suspension`: `good` → [0] (preview truncated)

The `vari-grip` mapping (`bird-nest`→0, `good`→1, `rust`→2) proves that 0/1 cannot be interpreted globally as normal/anomalous.
