# Plan

- [x] Verify Google Drive ZIP identities and sizes.
- [x] Parse actual ZIP manifests and discover categories/splits/classes.
- [x] Fix Benchmark Core to the five shared asset categories.
- [x] Implement manifest-driven QA generation.
- [x] Implement deterministic leakage-safe one-shot references.
- [x] Generate/validate 20-sample review sets from the real manifests.
- [x] Strengthen validation and evaluation metrics.
- [x] Add confusion matrix and invalid-output handling.
- [x] Add SHA256 + JPEG-dimension overlap confirmation tool.
- [x] Produce structure/count/mapping/data-issue reports from the real manifests.
- [ ] Run local SHA256 overlap confirmation against the two large source ZIPs.
- [ ] Read `annotations/image_list_val.txt` contents with the text-preview option and document the exact line schema.
- [ ] Next stage only: run one LVLM on the 20-sample QA set before any large inference.
