# Plan

## Requirement Delta — DINOv2 Retrieved 1-shot+

### Goal

Add a leakage-safe `1-shot+` Aligned Core regime that retrieves a query-specific same-asset train-normal reference with `facebook/dinov2-base`, while preserving every existing baseline and evaluation contract.

### Scope

#### In Scope

- Reuse the existing fixed one-shot Aligned Core QA as the immutable source of query IDs, images, GT answers, and audit comparison references.
- Build DINOv2 embeddings for the five Aligned Core assets, cache them, and select deterministic cosine Top-1 references per query.
- Emit separate `one_shot_plus.jsonl`, retrieval audit CSV, balanced 20–50 row smoke QA, predictions, metrics, and result directories.
- Connect the retrieved regime to Qwen3.5-4B, InternVL3.5-8B, and MiniCPM-V-4.5 only.
- Report overall, per-asset, and asset-macro anomaly metrics, including confusion counts and fixed-to-retrieved yoke-suspension FP change.

#### Out of Scope

- Changing or regenerating existing zero-shot, fixed one-shot, structured, Luxia, or LLaVA baselines.
- Running the costly 1,378-row LVLM inference without explicit user action.
- Adding retrieval variants beyond DINOv2 cosine Top-1.

### Current State

- The authoritative current checkout is the nested repository at `main@ce8d51d`; the outer checkout is stale and contains this checkout as an untracked directory.
- Fixed references are selected by `select_reference_by_asset()` from same-asset train normals after excluding every candidate whose `size:CRC32` fingerprint occurs anywhere in the unsupervised test split.
- The full runner and adapters already preserve `[normal reference, query]` image order, but the runner, evaluator, and diagnostics only know `zero_shot`, `one_shot`, and `structured`.
- Both source ZIPs are available locally; no raw data is committed.

### Decisions

- Extract the existing candidate eligibility logic into one shared helper; do not reuse the weaker, non-production helper in `qa_utils.py`.
- Use the aligned fixed one-shot QA as the builder input, replacing only reference fields and adding retrieval metadata. Keep fixed reference paths for audit only.
- Cache normalized embeddings in a non-pickle NPZ schema keyed by the immutable DINOv2 commit and `(path, asset, content fingerprint)`; reuse only exact matches.
- Resolve cosine ties by lexicographically sorted reference path so results are deterministic.
- Extend the existing HF runner through an opt-in `one_shot_plus` experiment mode. Its legacy default remains unchanged; plus mode defaults to a balanced smoke sample and requires an explicit full-run flag for all 1,378 rows.
- Keep all existing model adapters and LLaVA registrations intact; reject LLaVA only in the plus experiment.
- Preserve the evaluator's existing invalid-output policy while exposing its confusion counts explicitly.

### Architecture / Flow

`source ZIP + manifest + aligned fixed QA` → shared leakage-safe candidates → cached DINOv2 CLS embeddings → same-asset cosine Top-1 → `one_shot_plus.jsonl` + retrieval CSV → balanced smoke QA → existing HF adapters → additive evaluation/comparison reports

### Implementation Steps

- [x] Add the shared candidate filter and regression tests for the fixed selector.
- [x] Implement lazy DINOv2 embedding, safe incremental cache, deterministic batched cosine retrieval, QA construction, and retrieval validation as model-download-free testable units.
- [x] Add a builder CLI that reads the ZIP/manifest/fixed QA, writes full and smoke QA plus the audit CSV, and never overwrites baseline artifacts.
- [x] Add opt-in runner support for the three requested models, distinct checkpoints, selected-QA snapshots, smoke/full guards, and retrieved-reference-aware resume checks.
- [x] Extend evaluation and offline diagnostics with `one_shot_plus`, confusion metrics, asset-macro metrics, per-asset fixed/retrieved deltas, and yoke FP comparison.
- [x] Add a dedicated Colab notebook and concise README commands while leaving existing notebooks unchanged.
- [x] Run targeted tests, the complete local suite, compile/static notebook checks, manifest-backed leakage validation, and an actual retrieval smoke/full build. Do not run full LVLM inference.

### Definition of Done

- Every one of the 1,378 Aligned Core queries can receive a deterministic query-specific train-normal reference from the same asset.
- No anomalous, non-train, same-image, or duplicate-test candidate can be selected.
- Full QA, balanced smoke QA, cache, and retrieval audit CSV are produced separately from existing artifacts.
- Qwen, InternVL, and MiniCPM can run retrieved smoke inference through existing adapters; LLaVA and all legacy paths remain unchanged.
- Existing evaluation outputs remain readable and additive outputs provide overall/per-asset/asset-macro metrics plus fixed-to-retrieved yoke FP change.
- All automated regression tests pass; actual validations are reported separately from commands that were not run.

### Verification

- Pure unit tests use synthetic embeddings and synthetic ZIP fixtures; they never download DINOv2.
- Builder validation checks required metadata, exact query/GT preservation, candidate provenance, and global test-fingerprint leakage.
- Runner tests check legacy compatibility, three-model allowlist, reference-before-query ordering, smoke determinism, explicit full guard, and distinct prediction reuse.
- Evaluator/analyzer tests lock confusion semantics, unweighted asset macro, separate result-root joins, and yoke FP deltas.
- Final regression: `python -m pytest -v` in the authoritative nested checkout.

### Completion Evidence

- Pinned DINOv2 retrieval built 1,378 rows across the exact five assets from 10,383 cached embeddings; the balanced smoke QA contains 25 rows.
- Manifest-backed QA validation passed with zero errors/warnings and exact zero-shot/retrieved ID, image, asset, GT, and order alignment.
- The pickle-free cache is reusable without re-embedding and its SHA-256 is recorded in every QA/audit row.
- The complete local suite passes (123 tests); source compilation, CLI help, notebook compilation, and `git diff --check` pass.
- LVLM smoke/full inference was not run locally because the available 6 GB GPU is below the runner's 20 GB requirement; the guarded L4/A100 Colab path is ready.

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
- [x] Add initial open-source baseline families: Qwen, InternVL, MiniCPM, and LLaVA.
- [x] Replace the GPU-only Colab runner with a hybrid baseline runner.
- [x] Route closed baselines through official APIs: OpenAI GPT and Google Gemini.
- [x] Keep open-source baselines on direct Hugging Face GPU inference.
- [x] Use `main` in Colab and support private-repo checkout through a read-only `GITHUB_TOKEN` secret.
- [x] Add OpenAI/Gemini secret handling, retries, incremental predictions, unified evaluation, leaderboard, review CSV, and run metadata.
- [x] Skip direct HF models below 20 GB VRAM instead of silently changing the research condition with 4-bit quantization.
- [x] Extend notebook static tests for API/HF separation, secrets, model registry, and low-VRAM behavior.
- [x] Remove Google Drive mounting from the Colab experiment path; read source ZIPs directly from `/content` and download the final result ZIP locally.
- [ ] Add `GITHUB_TOKEN`, `OPENAI_API_KEY`, and `GEMINI_API_KEY` to Colab Secrets.
- [ ] Upload `unsupervised_anomaly_detection.zip` and `supervised_fault_classification.zip` to the Colab Files panel (`/content`).
- [ ] Execute GPT/Gemini smoke baselines: Zero-shot 20 / One-shot 20 / Structured 20.
- [ ] Execute Qwen/InternVL/MiniCPM/LLaVA smoke baselines on an L4/A100-class rented GPU.
- [ ] Review prediction validity, metrics, latency, and model-specific failures.
- [ ] After smoke-test review, expand the same registry/contract to full 6,481 / 6,417 inference.
- [ ] Analyze failure modes and design the first performance-improvement method.
