# Vision Expert Smoke Runtime Notes

The Colab notebook is designed to be rerunnable after interrupted sessions.

- The local `/content/vision_expert_workspace` is reused only when its QA identity, manifest identity, query order, training pools, and extracted image files all validate.
- A stale, partial, or mismatched local workspace is removed and rebuilt automatically.
- Workspace output is built in a temporary sibling directory and moved into place only after validation.
- The pinned official AnomalyCLIP checkout receives a deterministic compatibility-only patch for modern Python packages. The patch replaces removed `pkg_resources` packaging imports, imports `urllib.request` explicitly, and removes an unused undeclared `thop` profiling import. Model weights and inference logic are unchanged.
- AnomalyCLIP's base model download uses `/content/anomalyclip_model_cache` explicitly.

Validation performed before merge:

- complete repository test suite
- source, script, test, and all notebook code-cell compilation
- exact notebook dependency installation
- anomalib `Tabular`, `Patchcore`, `Dinomaly`, and `Engine` runtime API checks
- pinned AnomalyCLIP source checkout, checkpoint presence, compatibility patch, and imports

Actual GPU training and inference on the private Google Drive ZIP files remain a Colab execution step and are not claimed by these checks.
