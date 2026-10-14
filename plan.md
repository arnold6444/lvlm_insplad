# Plan

## Requirement Delta — PR56 review correction before gold (2026-10-14)

- Fix: fetch an immutable execution commit and verify clean exact HEAD on fresh and cached Colab runtimes, before inference; moving main is not a gold lock.
- Reporting: Q1 balanced accuracy, precision/recall, confusion counts/support; Q2 defect balanced accuracy; asset-macro table. Gold Q3 CI scope must say gold; preserve dev helper default.
- Recovery: separate CPU-only aggregation from saved QA/manifest/raw predictions; verify identities, prompt/choice/coverage, reconstruct only the declared routes, exclude failed/invalid dependencies. Keep original prediction identities, metadata and reports; write aggregation_v2 with distinct source and aggregation provenance.
- Preserve: no prompt/routing/model/candidate changes, no new GPU run, original gold QA/results and local guards retained. Existing PR56 raw predictions remain reportable without inference.
- Pin strategy: commit inference/report fixes first; notebook-only follow-up pins that reachable full SHA, avoiding a self-referential commit. New GPU run uses a separate v2 folder; CPU notebook discovers saved v1/v2 runs.
- Verify: hand-calculated required metrics, actual Git checkout with main advancement/wrong HEAD/dirty tree, zero-inference recomputation and raw-byte preservation, corrupt/missing/failed cache gates, notebook code/display/prepare, legacy default scope.
- Completed: immutable execution/report commit792c9ea, full-SHA fetch/detached checkout and clean-HEAD checks in both notebooks, CPU discovery/reporting, required metrics and gold scopes. Review/new/legacy tests46 passed; added stale-report error gate and notebook integration checks3 passed. No GPU inference.
- Remaining: correction PR merge/latest-folder sync, then user evaluates once or repairs complete saved predictions on CPU.

## Requirement Delta — locked Q1/Q2/Q3 candidate gold Colab (2026-10-14)

- Goal: prepare and merge an all-run gold notebook for Q1 baseline unchanged + revised fresh-definition Q2 + previous presence/optional-position Q3. User authorizes push/merge; actual GPU execution is in user Colab.
- Preserve: original QA/results and local guard edits; per-question independence; ref-first images, greedy unquantized InternVL; gold246 remains FROZEN. No manual labels, automatic selection, expert integration or new-Q3-review route.
- Lock: InternVL only, dev240 posthoc choice stated explicitly. Same diffdate retrieved normals as dev. Reuse verified stored gold Q1/Q2 predictions when available; otherwise generate the direct baseline in the new folder, then hold its Q1 fixed. Baseline and previous-agent are predeclared controls, never gold-based selection.
- Q3: generate separate 9-region+none auxiliary questions for all246, score image-label presence only, spatial accuracy null. Original gold42 four-option QA untouched; do not merge its localization accuracy with this proxy. Gold has already been explored, not a new unseen holdout.
- Flow: preflight QA/config/refs/archive identities → immutable protocol lock → validated baseline cache or new direct calls → revised Q2 and legacy Q3 → dependency-complete comparisons, per-asset/defect support and paired CI → status and result ZIP. Invalid/error calls retry; failed policies excluded; complete flag requires every planned policy, coverage and valid calls. No frozen_config/selection/publish.
- Review: reuse existing prompt/metric/extraction/adapter utilities instead of generalizing dev runner or invoking all seven dev ablations. Code/config/image/prompt identities bind resume; persistent prediction tails survive interruption. Test failures, invalids, model-load exceptions, independent prompts, tampering and Q1 preservation.
- Steps: runner/config → Colab/protocol documentation → CPU fixtures + legacy prompt/adapter tests + notebook compile/display → PR merge → fresh-folder fast-forward with local changes and research hashes preserved.
- Done: runnable merged notebook with automatic Drive asset discovery/reference preparation, explicit stop conditions, original result preservation, meaningful fixture verification and clear user filename/runtime instructions. No local LVLM performance claim.
- Completed: committed pre-gold config, candidate runner, automatic Colab, protocol and README. Twelve candidate tests pass (cached/new baseline, Q1 preservation, GT/answer independence, failures/invalids/load failures, resume/tail, immutable input and notebook preparation/display). Existing related runner/prompt/adapter tests: 52 passed. Notebook code cells compile, CLI help and diff checks pass. Actual GPU gold inference is not run locally.
- Publication: user explicitly authorizes PR merge after verification; latest-folder sync must preserve local guard edits and research artifacts.
- Remaining: user runs the prepared notebook once and returns ZIP; actual GPU gold results are pending.

## Requirement Delta — Q2 rust / Q3 missed-defect refinement (2026-10-13)

- Goal: diagnose the three rust regressions and five Q3 false-none cases, implement an independently testable refinement, and merge an all-run larger-dev Colab. User explicitly authorizes merge. Keep Q1, original results and gold unchanged; no annotations.
- Evidence: original responses parse correctly, errors/invalids zero. Q3 four misses exit at presence-none, one presence-positive becomes position-none. Seven unique original image/reference pairs inspected locally. Rust has dark/rough metal and background twigs; glass-disc loss can leave caps present; foreign twigs can blend into background. These are visual/prompt hypotheses, not proven model reasoning (raw outputs contain choices only).
- Decisions: default full dev240, with design40 (same seed42 original subset) and additional-dev200 separately scored; the latter was excluded from this diagnosis but is still development data, not a new formal holdout. Q2 exact old definitions without prior-answer anchoring are an ablation; balanced updated definitions explicitly cover dark/rough corrosion, missing glass/discs, hardware-vs-background material. Q3 refined presence and position both run for every image; disagreement gets own-Q3 review, invalid falls back to valid own position/direct, never other question answers. No forced region or rust preservation rule.
- Conditions: baseline, previous agent, fresh old definitions, revised Q2 only, revised Q2 + always-position, revised Q2 + Q3 independent review, matched-call reask. Same inputs/options/greedy adapters; no change to Q1 or source QA. Old runner defaults and resume identities remain compatible.
- Flow: shared legacy calls + fresh-definition/cue tools + full-frame Q3 presence/position + optional own-Q3 disagreement review → complete-policy metrics/CI by all/design/additional cohorts. Always-none/region controls and Q3 spatial accuracy null retained. New dedicated output folder, raw/prompts/checkpoints, image/input identity guards, failure exclusion/retry, no selection/freeze/gold.
- Review: use the existing runner/cache/report implementation rather than copying a second engine; the opt-in concrete refinement supplies prompts/calls/cohorts. Do not increase image/tile resolution in this experiment, to keep a separate cause from instruction/routing changes. No per-case rules, answer leaks, or manual spatial ground truth.
- Done: diagnostic report with case-level observations/limits; executable no-manual Colab for dev240, main paired comparisons + cohort metrics/CI/coverage; Q1 independence, disagreement/invalid/error/resume/preservation tests; old protocol regressions; notebook compilation/display; PR merge and fresh-folder sync preserving existing local changes.
- Completed: case-level diagnostic/protocol report; opt-in revised prompts and always-position/disagreement-review flow; all-run Colab dev240, cohort metrics/CI/support and matched controls. Related new/legacy/prompt/adapter tests: 52 passed; notebook compilation and fixture display pass. Actual QA partition: full240 (125 normal/115 anomalous), design40 (24/16), additional200 (101/99). Fourteen inspected input hashes match the original manifest. Actual old archive resume: 232 reused, zero new calls and no model load. CLI help and diff checks pass.
- Remaining: PR/merge and fresh-folder sync, then user larger-dev inference. Actual GPU improvement/time and spatial localization remain unverified.

## Requirement Delta — Q1 fixed, Q2/Q3 independent dev agent (2026-10-13)

- Goal: a no-manual-work Colab comparing Q2 definition and Q3 independent presence/position tools while retaining Q1 stage-1 answers exactly. User authorizes publication/merge; existing outputs and gold remain untouched.
- Scope: deterministic dev40 by default, InternVL (Qwen optional). Q2 uses the existing own-question define tool; Q3 sees only its images/question, never Q1/Q2 answers. The baseline and equal-call reask control use the same images and decoding.
- Q3 contract: generated 9-region+none options, shuffled without GT; normal rows have a none answer, anomalous rows have no location answer. Score only none-vs-region presence, normal false-region rate, anomaly false-none rate, invalids, and presence-only joint correctness. Location accuracy is null, never inferred from defect labels. Always-none/always-region baselines expose trivial policies.
- Flow: reuse Q1/Q2 stage1 → Q2 reask/define → fresh Q3 direct → independent Q3 presence → optional whole-image position tool → final answers → evaluation. A matched reask control uses the same additional call count per image as the Q3 agent. No automatic setting selection/frozen config or gold run.
- Preserve: question independence, ref-first images, greedy HF/no quantization, immutable original QA/results, distinct output folder, 20-call checkpoints. Resume binds prompts, own answers, input image hashes and model specs. Failures persist, retry and exclude affected policies from comparisons.
- Done: executable all-run notebook with no annotation/UI interaction, fresh metrics/CI and raw responses, fixture independence/preservation/retry/partial-label checks, notebook compilation, PR merge and latest-folder sync. Actual LVLM execution remains in user Colab.
- Completed: independent runner, all-run Colab notebook and protocol documentation. Targeted Q2/Q3, prompt-diagnostic, legacy independent-agent and HF-adapter checks: 43 passed. Notebook code cells compile and the results display cell executes against fixture outputs. CLI help and diff check pass. Actual dev QA selects 40 images (24 normal / 16 anomalous); positive location answers remain null.
- Publication: PR #54 squash merged; latest folder synchronized to main@7edee7c (2026-10-13). Existing local guard edits preserved by SHA256 verification. Synced-folder new-runner tests: 12 passed.
- Colab result audited (2026-10-13): InternVL dev40/L4, 232 calls, zero errors/invalids, Q1 unchanged. Q2 F1 .488636→.696936 (baseline CI crosses zero), Q3 presence BA .5625→.760417 (CI +.041648..+.333333); normal false-region21→4/24, anomalous false-none0→5/16, rust Q2 correct8→5/10. Raw outputs and metrics/CI independently checked; archive preserved in reports/q2_q3_evaluation_2026-10-13/a235a8c4ddf8.
- Remaining: investigate Q2 rust regressions and Q3 false-none (four presence / one position), then larger dev validation. Anomalous localization remains unverified; no configuration freeze, gold execution or result publication.

## Requirement Delta — dev part-crop diagnostic (2026-10-12)

- Goal: test whether showing matching physical part regions improves Q1/Q2 before investing in automatic ROI selection.
- Scope: same deterministic dev 40 images / 80 questions as the prompt diagnosis; a paired normal/target box UI; a full-image vs part-crop runner and Colab notebook; InternVL default, 160 calls maximum. User authorized PR publication and merge.
- Preserve: original QA, predictions, gold and frozen configuration; question independence, baseline prompt, reference-first image order, no quantization. Separate `gold_v2_part_crop_diagnostic_v1` output. No gold run or automatic setting selection.
- Decisions: draw one physical inspection-part box on both images, not defect-only boxes; hide labels, source paths and model answers; mark unavailable pairs explicitly and exclude them from both conditions while reporting coverage. This is a manually assisted diagnostic, not automatic deployment evidence.
- Steps: implement annotation/store and resume guards → paired inference/report → Colab UI → fixture, browser, notebook and main-baseline checks → PR/merge → synchronize latest folder.
- Done when: annotation persists each pair and resumes; wrong/missing/stale boxes block inference; paired inputs and fresh baseline are identical across conditions except cropping; raw outputs/prompts/image hashes retained; errors retried and excluded from comparison; notebook opens from merged main.
- Verification: synthetic images/ZIPs, failure/retry and preserved-result tests, actual UI drag/save/reset/resume preview, notebook compile and target adapter/agent tests. GPU inference remains for the user's Colab.
- Completed: annotation/store, paired inference, Colab notebook, 35 diagnostic/agent/adapter tests; browser drag/save/unavailable/reset/resume and 390px layout verified with the actual Python storage backend; notebook compilation and CLI help passed. GPU inference not run locally.
- Publication: PR #53 squash merged; latest folder synchronized to main@9e0266c (2026-10-12).
- Remaining: user physical-part annotations and Colab inference; no automatic localization/gold claim.

## Plan — expert-evidence verification agent (2026-10-12, not implemented)

**Goal.** Vision-expert evidence (score, ROI, heatmap, combinations) is often wrong. Test whether checking each piece
of evidence, per question, before the LVLM relies on it improves Q1/Q2 accuracy, and how often the check accepts
wrong evidence. Same contract as `docs/verify_agent_independent.md`: each question on its own, contradiction only
reported, choose on dev, gold once.

**Why.** Gold, 4-model mean Q1 F1: normal reference .758 vs PatchCore score .674 / heatmap .502 / ROI .402;
ref+score has the highest Q1-Q2 contradiction (.430). Maps are min-max normalised per image
(`vision_expert.py:59-69`), so every normal image shows a hot spot; an earlier analysis put the ROI-centre hit rate
on the 20 gold rust boxes at .40 (PatchCore) / .30 (AnomalyCLIP) — PR #50 re-measures it. The team's vari-grip
report also lost 8.1%p when the map was added (CI excludes 0).

**Code constraints to fix before any dev run.**
1. Leakage: expert training normals exclude only unsupervised-test fingerprints (`dataset.py:265`). Every dev normal
   is in that pool (`results/latest/gold_v2_dev/build_summary.json`), so an unchanged rebuild puts many dev normal
   targets in the PatchCore / ReConPatch-lite memory banks (up to 200 of 238–595 per asset) or in Dinomaly's
   training set, making them look normal. AnomalyCLIP (zero-shot, VisA checkpoint) is unaffected.
2. Transductive percentile: `add_score_percentiles` ranks within the evaluated cohort
   (`run_gold_v2_vision_expert_full.py:79-87`). Use a train-normal calibration instead (rank of the raw score among
   held-out same-asset train normals), the same on dev and gold.
3. Per-image heatmap normalisation: the agent's evidence uses a fixed per-asset scale from held-out train normals.
4. The gold expert path assumes 246 cores, the `master_qa_*` glob and unsupervised-ZIP query paths; dev images are
   supervised-train paths (dev defects are not in the unsupervised ZIP) → needs a dev workspace builder.
5. The heatmap prompt still mentions a yellow box (`run_gold_v2_vision_expert_full.py:109-111`); new prompts must not.

**Decisions to confirm (recommendation first).**
- D1 experts: PatchCore (best expert so far) + AnomalyCLIP (other family, no InsPLAD normals → independent check).
- D2 evidence: score and ROI (peak crop); heatmap overlay only as an ablation.
- D3 check, per question: the question's own text and options asked about the evidence (Q1 "is the marked region
  normal or anomalous?", Q2 "which condition best describes the marked region?"; score: score sentence +
  [reference, target]). Accepted when the answer agrees with the evidence's claim, rejected otherwise. No shared
  "is there a defect" call across questions.
- D4 final rule (fixed): majority of {stage-1 answer} ∪ {answers backed by accepted evidence}; tie → stage 1.
  An LLM orchestrator is only an ablation.
- D5 dev models: InternVL3.5-8B + Qwen3.5-4B.

**Flow per question Qk** (Q1, Q2 on dev; Q3 auxiliary on gold)
```
stage 1: [diffdate normal reference, target] -> a0             (reuse dev baseline)
for e in {PatchCore score, PatchCore ROI, AnomalyCLIP score}:
    check(Qk, e) -> accept / reject, plus the evidence-backed answer
final(Qk) = fixed rule over a0 and accepted answers
after all questions: Q1-Q2 (-Q3) contradiction, reported only
```

**Evaluation.**
- Evidence level (is the check trustworthy): score evidence is correct iff (calibrated score above the train-normal
  threshold) == GT anomalous → accept rate on correct vs wrong evidence. ROI on normal images is always wrong →
  rejection rate. ROI on anomalous dev images has no location label → accept rate only.
- Answer level: Q1 F1, Q2 macro-F1, per-question accuracy, FDR, contradiction new/resolved, calls, latency.
- Dev comparisons: stage 1 / independent verifier / evidence without the check / evidence with the check / ROI
  location controls (centre crop = existing `zoom`, random crop) / compute-matched re-ask.
- Adoption rule (set before the run): adopt only if the agent beats both the independent verifier and the
  compute-matched control on Q1 F1 or Q2 macro-F1 with the paired-bootstrap CI excluding 0; otherwise report it as
  a negative result.

**Steps.**
- [ ] P0 (CPU, no LVLM): merge PR #50 and run the heatmap check on gold — analysis only, chooses nothing.
- [ ] P1 code: dev expert workspace from the supervised ZIP; drop every dev and gold target fingerprint from the
  training normals; hold out train normals for calibration. Tests: no target fingerprint among training normals,
  calibration does not depend on the evaluated cohort.
- [ ] P1 run (Colab L4, experts only): PatchCore per asset + AnomalyCLIP on dev 240 and on the held-out normals;
  report dev expert AUROC per asset, with and without same-flight-date normals in the bank. Gold evidence gets the
  same calibration (decide in P1: score held-out normals with the existing models or refit).
- [ ] P2 code: extend `run_gold_v2_verify_agent.py` with evidence tools and controls so the independence test,
  `answers.csv`, frozen config and cost columns carry over; add an evidence-level metrics file.
- [ ] P2 run (dev): ~480 questions × 3 evidence × 2 models ≈ 2,900 check calls + ~1,900 control calls, ~2–3 h L4.
- [ ] P3: choose on dev → `frozen_config.json` → gold once (needs calibrated gold evidence and the gold re-run's
  stage 1).

**Done when.** Dev expert outputs are leakage-safe and calibrated (tests); evidence- and answer-level tables on dev
with controls; one frozen config; one gold run; the independence test passes for the new tools.

**Unknowns.** Expert runtimes are undocumented; held-out calibration size per asset is set in P1.

## Requirement Delta — question-independent verification agent (2026-10-11)

- **Goal**: test whether re-checking each question on its own improves accuracy, and measure the contradictions that
  remain. Protocol: `docs/verify_agent_independent.md`.
- **Changed**: `scripts/run_gold_v2_verify_agent.py` rewritten (PR #49 routed on Q1-Q2 contradiction and rewrote
  Q1/Q2/Q3 from one Q2 re-check). Now per question: prompt, routing, tool and final answer use only that question's
  own answers; contradiction is computed afterwards and reported only. Dev notebook updated; gold notebook added
  (runs only the dev-frozen variant).
- **Dropped**: `rule_q1_first` and the contradiction trigger.
- **Kept**: stage 1 = `normal_ref_retrieved_diffdate`, dev for selection, gold once; existing results untouched.
- **Done when**: independence test passes for Q1/Q2/Q3 (done locally); Colab runs below are done by the user.
- **Remaining (Colab, not run)**: dev baselines `normal_ref_retrieved_diffdate` → dev agent → publish
  `frozen_config.json` → gold re-run (gold refs + stage 1) → gold agent once. Q3 has no dev labels.

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

> **Correction (2026-10-07):** the nested-checkout note under Current State is stale — the working checkout is `Desktop\lvlm_insplad_fresh` tracking `origin/main`. On `origin/main` (`4e3ebc3`) the suite has 187 tests: with `--basetemp=.pytest-tmp`, 185 pass and 2 fail in `tests/test_anomalyclip_compat.py` (the compat patch rejects an `import urllib.request` fragment). Without `--basetemp`, 56 tests error on Windows temp-dir permissions.

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
