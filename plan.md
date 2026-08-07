# Plan

- [x] Verify authoritative Google Drive ZIP identities and sizes.
- [x] Parse actual ZIP manifests and discover categories/splits/classes.
- [x] Verify supervised annotation line schema from local ZIP text previews.
- [x] Confirm all 8,990 cross-dataset overlap candidates with SHA256 and dimensions.
- [x] Expand Track A to all 17 unsupervised assets on official test.
- [x] Keep Track B on the 5 supervised assets using official val because no test exists.
- [x] Define the Aligned Core on exact same images across Track A/B evaluation splits.
- [x] Exclude 6 exact-image binary-GT conflicts from the clean Aligned Core.
- [x] Implement deterministic leakage-safe one-shot references for all 17 Track A assets.
- [x] Generate and validate Track A/Track B QA from the real manifests.
- [x] Generate exact-overlap/core reports and 20-sample review sets.
- [x] Strengthen evaluation metrics, invalid-output handling, and confusion matrix.
- [x] Run fixture-based regression tests.
- [ ] Next stage: run one Hugging Face LVLM on the 20-sample sets and inspect output/QA behavior.
- [ ] Only after the smoke test: decide whether to run full 6,481 / 6,417 inference.
