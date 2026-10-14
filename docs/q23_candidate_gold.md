# Q1 fixed, revised Q2 and previous Q3 gold candidate

## Run

Open `notebooks/lvlm_gold_v2_q23_candidate_colab.ipynb` from merged main in a **fresh L4/A100 Colab runtime**. Allow the existing `GITHUB_TOKEN` Secret, choose **Runtime → Run all**, authorize Drive, then return `gold_q23_candidate_results.zip`. No boxes, labels, manual configuration or GitHub result publication. Dependencies use the existing pinned Transformers/Accelerate versions. No quantization; VRAM must be at least 20 GiB.

Default output: `MyDrive/lvlm_insplad_results/gold_v2_q23_candidate_v1/`. Original gold QA, dev answers and other result folders are read-only. Repeating the inference cell resumes this same run; it is not an independent repetition.

## Pre-gold lock

`configs/q1_fixed_q2_cues_q3_legacy_gold_v1.json` commits the InternVL-only candidate before gold. No automatic ranking or `frozen_config.json` is produced. `protocol_lock.json` binds inputs/code/images before inference and is explicitly **not** a completion or successful-selection file. `run_status.json.complete` requires all three prespecified policies with complete Q1/Q2/Q3 answer coverage, zero inference/load errors and zero invalid dependency outputs. Failed policies do not enter leaderboard or paired comparisons, even if a valid fallback final exists.

The candidate was chosen **posthoc** from PR55 dev240. Additional200: revised Q2 macro-F1 .697842 vs baseline .514010 (positive paired CI); previous .653941 (CI crosses zero). Previous Q3 missed 15/99 anomalies, revised Q3 missed 29/99 (increase CI positive). This recommends retaining legacy Q3, not adopting new review. Hybrid stored-answer joint .635 is posthoc, not a new pipeline inference. Gold246 already has exploratory use; this evaluation is not a newly untouched holdout.

| Policy | Q1 | Q2 | Q3 auxiliary |
|---|---|---|---|
| baseline | direct gold baseline fixed | same direct baseline | direct Q3 |
| previous_agent | identical Q1 | own prior answer plus original definitions | legacy presence then optional position |
| candidate | identical Q1 | PR55 fresh revised definitions, no prior answer | same legacy Q3 |

Controls are fixed comparisons, not new gold tuning. Candidate-vs-previous shares Q3 exactly; its joint change comes from Q2. No expert, reask-Q1, refined Q3, Qwen, or manual crop integration.

## Input and baseline preparation

- Read frozen `lvlm_insplad_qa/gold_v2_final`, require original246 paired Q1/Q2 rows. Original42 localization questions remain intact.
- Use same-asset unsupervised-train normal references with same-flight dates excluded. Reuse `gold_v2_main_rerun/refs_gold/retrieved_refs.csv` when available, otherwise build refs in the **new** run folder with the existing DINOv2 retrieval script. This is normal-reference search, not vision-expert defect evidence.
- Reuse valid stored `gold_v2_main_rerun/predictions/internvl3_5_8b/normal_ref_retrieved_diffdate.jsonl` when available. Validate model, condition, prompt version, core/type, reference mapping and parsed choice. Missing/error/invalid baseline rows are freshly generated only in the new folder, without altering the older source. A mismatched source identity stops the run.
- If no baseline exists, generate Q1/Q2 direct baseline first (492 calls), then hold Q1 fixed. **Zero candidate Q1 inference** means no recheck/change; it does not mean zero initial baseline calls when one is missing.
- Candidate Q2, previous Q2, direct Q3, presence Q3 and optional position Q3 total **984–1,230** new shared tool calls. Missing initial baseline adds at most492 calls. With the PR55 observed cost as a rough guide, allow approximately30–70 minutes plus setup/retrieval; this notebook's actual runtime is unmeasured.

## Q3 contract

Generate **separate auxiliary** nine-region-plus-none questions on all246 images with deterministic option shuffle independent of GT. Legacy presence controls whether the same Q3 full-frame position prompt runs. Negative presence chooses none; invalid output falls back to own direct Q3, but marks the policy ineligible until retried. Q1/Q2 answers/labels do not influence Q3 routing. Likewise revised Q2 never receives Q1/Q3 or prior Q2 answers.

Evaluate normal none accuracy, normal false-region rate, anomaly region recall/false-none rate, balanced presence accuracy and Q1/Q2/Q3-**presence** joint. A wrong location still counts as region-present: `q3_location_accuracy=null`, zero anomalous spatial labels. Always-none and always-region controls expose trivial policies. Do **not** substitute the original42 four-option center-biased Q3 accuracy into these tables or claim actual localization improved.

## Recovery and evidence

Raw per-step prompts, reference-first image paths, outputs, parsed choices, errors, latency and source are saved every20 steps. Resume binds config/code/prompt/model/QA/reference and image byte identities, preserves unprocessed checkpoint tails, reuses valid successes and retries only failed/invalid/newly required calls. Changed inputs require a new output folder; no previous results overwritten.

Results: `leaderboard.csv`, `answers.csv`, `metrics_by_asset.csv`, `q2_by_defect.csv`, `significance.csv` (paired bootstrap2,000), `q3_constant_baselines.csv`, partial-Q3 and Q1/Q2 QA snapshots, image manifest, metadata/status/protocol lock and raw step predictions. Baseline imports, newly generated baseline calls, tool calls and reused calls are separate; shared counts must not be summed across policies.

Stop on non-FROZEN QA, wrong image/QA/ref/schema/model identity, unsafe archive paths, inconsistent resume, invalid/errors or missing coverage. Retry the inference cell for transient failures. Inspect the summary before treating the experiment as complete. No further setting search after reading gold.
