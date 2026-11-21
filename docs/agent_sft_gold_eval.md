# Gold scoring of the fine-tuned agent (protocol fixed 2026-11-17, controls added 2026-11-19, run once on 2026-11-20)

Everything in this file except the "Result" section at the end was written before the run and is unchanged.

Notebook: `notebooks/lvlm_agent_sft_lora_human_boxes_gold_colab.ipynb`. Pick an L4 or A100 runtime, then **Run all**
once. It trains the second LoRA run ([agent_sft_lora.md](agent_sft_lora.md), `sft_v2_human`), or reuses it when it
is already on Drive, and then scores gold v2 in the same execution. The user can run Colab only once more, so nothing is chosen between the steps: every row
below is fixed here, before any gold call, and every row that runs is reported.

## State on Drive (read on 2026-11-19, read-only)

- The second run is already trained. The earlier notebook (source `fb5b9be`) ran on an L4 on 2026-11-17:
  4,791 training and 249 validation samples, 599 optimizer steps, validation loss .779 to .143. Its data folder and
  adapter are on Drive.
- Its dev240 evaluation (four conditions) stopped midway on 2026-11-17 and has no leaderboard.
- No gold scoring exists, and no gold reference file was found (`gold_v2_matched_call_v1` has no `refs_gold`), so
  the notebook builds the references itself with `build_retrieved_refs.py`. Rehearsed locally on the real gold photo
  names and both real ZIP manifests with fake embeddings: 246 rows, every reference a different-date train normal
  of the same asset, accepted by the runner.
- Present and as the notebook expects: frozen gold QA (`master_qa_492.jsonl`, `FROZEN`, 246), both source ZIPs at the
  top of My Drive, the first run's adapter and training list.
- So this notebook skips data building and training: both scripts compare settings and data hashes, not source
  commits (checked with the new source on the real first-run folders: "already built", "already trained").
  What remains is scoring, about 6 to 7 hours on an L4 with the controls.

## Why

The paper's fine-tuning section has dev240 results only. Dev240 comes from the same supervised-train pool and dates
as the training photos, and it has no location labels. Gold v2 is the only set with 133 located defects and with
polymer defects, so it is the only place where the three open questions can be answered:

1. Does the fine-tuned model keep its Q1/Q2 scores on photos that are not from the training pool?
2. Does it locate defects, or does it only match a constant "center" answer?
3. Does the learned crop point at the defect?

## What is scored

Runner: `scripts/run_agent_sft_gold_eval.py` (protocol `agent_sft_gold_eval_v1`).

- **Same agent as dev240.** Prompts, options, the strict JSON parser and the episode loop are
  `run_agent_tool_pilot`'s. Q3 is the pilot's nine-cell question with the same option shuffle; the legacy four-way
  gold Q3 is not used. No prompt, definition or threshold is new.
- **Questions.** Frozen gold v2: 246 photos, Q1 and Q2 from `master_qa_*.jsonl`, plus the nine-cell Q3. 738 question
  episodes per condition.
- **Reference.** Different-date DINOv2 Top-1 from unsupervised train normals, the rule the agent was trained and
  evaluated with. An existing gold reference file on Drive is reused only if it covers all 246 photos.
- **Conditions.** `direct_reference` (one call) and `adaptive_tools` (up to three calls) for every model.
- **Controls (added 2026-11-19, before any gold call).** For the primary model only: `fixed_evidence` (the fixed
  center-60% crop, one call) and `matched_reask` (as many calls as `adaptive_tools` made on that question, no tool).
  Without them a gain of `adaptive_tools` over the single call could not be told apart from "more calls" or from
  "any crop", and Colab cannot be run again to add them. They run as the last step, in the primary model's
  folder (`--controls`): the two main conditions are read back from the saved calls, the run signature does not
  change, and the two-condition tables are first copied to `main_conditions_before_controls/`.
- **Models, in run order.** The order is by importance, so an interrupted run keeps the main result.

| Order | Model | Why |
|---|---|---|
| 1 | LoRA `sft_v2_human` (human boxes, polymer added) | **primary** |
| 2 | Small classifier on the same labels (next section) | what the labels give without an LVLM; short |
| 3 | Qwen3.5-4B before training | same prompts, no training |
| 4 | dev240, LoRA `sft_v2_human`, the same two conditions, in a new folder | the dev row the paper's dev table lacks |
| 5 | LoRA `sft_v1` (automatic boxes), only if its adapter is on Drive | what the human boxes and polymer photos add |
| 6 | LoRA `sft_v2_human`, the two controls | needed only if tool use looks better than the single call |

The interrupted four-condition dev240 run cannot be resumed by this runner (a run is bound to its source), so it
is left untouched and its saved answers go into the result ZIP as `dev_lora_earlier_four_condition_run/`.
Dev240 gets no controls for the second model: it has no location labels, so the tool question is decided on gold.

## Small supervised baseline on the same labels (added 2026-11-19, user decision, before any gold call)

Runner: `scripts/run_sft_label_classifier_baseline.py` (protocol `sft_label_classifier_baseline_v1`). It answers the
question the fine-tuning section otherwise leaves open: what do the same few hundred labels give a small dedicated
classifier?

- **Training.** The 660 photos and labels of the second run's `sft_train_qa.jsonl`, nothing else. The class is the
  photo's Q2 answer (normal, rust, bird nest, missing cap); polymer `corrosão` photos carry the rust answer, as
  they do for the LoRA run.
- **Model.** Frozen DINOv2 CLS features (the encoder the reference search already uses) and
  `StandardScaler` + `LogisticRegression(max_iter=5000)`, scikit-learn defaults otherwise. One fit. No setting is
  tried against gold or dev240.
- **Scoring.** Gold v2 on both scopes and dev240, Q1 and Q2 only, with the agent runs' functions and invalid-output
  policy. Q1 is "anomalous" when the predicted class is not normal, so Q1 and Q2 cannot contradict each other.
  The runner is given the gold runner's training lists, so the frame-disjoint photos are the same 226.
- **What it is not.** It sees the target photo only (no reference photo, no asset name), and it gives no location.
  A higher Q1/Q2 score than the LoRA row would say that these labels are better spent on a classifier for
  detection and typing; it would say nothing about location or about answering in text.
- **Kept for later.** `features.npz` holds every feature (660 training, 246 gold, 240 dev), so another small
  classifier needs no GPU. Any such later variant is exploratory and is reported as such.
- It runs right after the primary model (about 15 minutes), because it is short and Colab cannot be run again.

## Scopes

- **Primary: frame-disjoint gold.** A gold photo is removed when it shares a drone frame (flight date + `DJI_nnnn`,
  `build_agent_sft_data.photo_key`) or its bytes (size + CRC32) with a photo of any fine-tuning set scored here.
  The same photos are removed for every model.
- **Secondary: all 246.** Reported next to it, never instead of it.
- Checked locally on the real gold labels, the real location set and the 660 second-run training photos
  (2026-11-17, no model call, dummy image bytes):

| | all | frame-disjoint |
|---|---|---|
| Photos (normal / defective) | 246 (107 / 139) | 226 (101 / 125) |
| Located defects | 133 | 119 |
| Constant "center" position hits | 72 | 62 |
| Fixed center-60% crop covers a box centre | 103 | 90 |
| Photos in the location joint (normal + located) | 240 | 220 |

  20 gold photos share a frame (6 normal, 14 defective: 4 missing-cap, 4 bird-nest, 5 vari-grip rust, 1 yoke rust);
  none shares bytes. The 72 of 133 matches the paper's constant baseline, which confirms that the position scoring
  here is the paper's.

## Metrics

- **Q1 / Q2 / coherence.** As on dev240: Q1 F1, balanced accuracy, precision/recall, TP/FP/TN/FN, invalid rate;
  Q2 macro-F1; joint accuracy; consistency error; false-defect rate; per asset and per defect.
- **Q3 presence.** Balanced accuracy, normal false-region rate, defect false-none rate.
- **Q3 position.** On located defects, a hit is a reported cell that is the centre cell of one of the photo's
  boxes (`score_gold_v2_q3_position`, the paper's rule). "No region" and an invalid answer are not hits.
  Reported with the constant-"center" hits and the best constant cell on the same photos.
- **Location joint.** Q1 and Q2 correct and, on a normal photo, "no region"; on a located defect, a position hit.
  The six defective photos without a box are not in it.
- **Crop.** Number of crop requests, distinct boxes, share of the most frequent box, and on located defects how
  often the Q3 crop contains a box centre, next to the same count for the fixed center-60% box.
- **Invalid outputs.** Unchanged policy: anomalous rows count as FN, normal rows as FP.
- **Intervals.** Paired bootstrap, 2,000 resamples, seed 42: `adaptive_tools` minus `direct_reference`,
  `adaptive_tools` minus each control that ran, and each condition's position hit rate minus the constant
  "center" answer. Differences between models (trained against untrained, second run against first) are computed
  afterwards from `per_image.csv`; they need no model call.

## What each outcome lets the paper say

Written before the run. No threshold is set for "keeps its score"; the gold numbers are stated next to dev240's.

| Claim | Allowed only if (primary scope, `sft_v2_human`, `direct_reference` unless noted) |
|---|---|
| Fine-tuning carries over to gold | report joint accuracy and Q2 macro-F1 as they are, with the base row and the dev240 row |
| The model locates defects | position hit rate minus constant "center" has a 95% CI above zero |
| Tool use helps | `adaptive_tools` minus `direct_reference` has a 95% CI above zero on that metric, **and** so has `adaptive_tools` minus `matched_reask`; if only the first holds, the gain is reported as "more calls" |
| Choosing the crop helps | `adaptive_tools` minus `fixed_evidence` has a 95% CI above zero on that metric |
| The crop points at the defect | the Q3 crop covers a box centre more often than the fixed center-60% box (90 of 119) |
| Human boxes and polymer photos help | stated as a difference between two runs that changed three things at once, never as one cause |
| Fine-tuning an LVLM beats a small classifier on these labels (Q1/Q2) | the LoRA row is above the `dinov2_linear` row on the primary scope; otherwise the paper says the classifier matches or beats it on detection and typing |

If position does not beat "center", the paper says location is unsolved. If the gold scores are far below dev240,
the paper says the dev240 scores reflected photos close to the training set.

## Limits

- One model, one training run per adapter, greedy decoding.
- Gold was used before by other methods, and its photos come from the same site and campaign as the training
  photos. Frame-disjoint removes shared drone frames, not shared dates or towers.
- The prompts and the retrieved reference differ from the paper's gold tables, so these rows are comparable with
  each other and with dev240, not with the baseline tables.
- The small classifier is one feature model and one linear classifier. It bounds the question from one side only:
  a tuned or fine-tuned vision model could do better than it.
- Estimated time on an L4 (from the first run's speed, not measured): gold about 4 h in total, dev240 1.5 h,
  the controls 1 h, the small classifier 15 min, and 6-7 h more only if training has to run.

## Safeguards for a single execution

- The first cell reads the GPU with `nvidia-smi` before Drive and pip, and stops with the attached GPU's name.
  The notebook metadata asks for an L4.
- The location set is embedded in the notebook and checked by SHA-256.
- Right after the training data is built, the gold runner runs with `--inputs-only`: gold QA, location set,
  references, ZIP members and the frame overlap are all checked before any training time is spent.
- Every step resumes from Drive: training checkpoints every 50 steps, saved calls are never repeated.
- `RUN_GOLD_BASE`, `RUN_DEV240`, `RUN_GOLD_SFT_V1` and `RUN_GOLD_CONTROLS` in the settings cell skip a step; a
  skipped row is absent.
- The result ZIP is rewritten on Drive after every step (written to a `.partial` file, then renamed), so a session
  that dies leaves the ZIP of the last finished step.
- Every command's output is appended to `agent_sft_gold_sft_v2_human/logs/run_*.log` on Drive, with `pip freeze`
  and `nvidia-smi`. An error is readable after the Colab tab is gone.
- Before each gold step the runner is asked (`--inputs-only`, exit code 3) whether its folder holds a run of other
  code, inputs or settings. Such a folder cannot be resumed; it is left as it is, zipped as `*_earlier_run/`, and
  the step runs in a new folder named after the notebook's commit. Without this, a notebook rebuilt after a fix
  would fail every gold step on a Drive that an earlier version had started on.
- The last cell flushes Drive before the runtime can go away.

## Outputs

`MyDrive/lvlm_insplad_results/agent_sft_gold_sft_v2_human/agent_sft_gold_eval_*/` per model: `leaderboard.csv`
(one row per scope and condition), `metrics_by_asset.csv`, `q2_by_defect.csv`, `position_by_defect.csv`,
`per_image.csv`, `significance.csv`, `frame_overlap.csv`, `run_metadata.json`, `run_status.json`, saved calls.
Everything is also in `agent_sft_sft_v2_human_gold_results.zip` on Drive.

### What the ZIP must hold so that no number needs a second GPU run (checked 2026-11-19)

| Needed for | In the ZIP |
|---|---|
| Any metric, rule or parser applied later | `*/episodes/<condition>/*.jsonl`: every call's prompt, raw output, error and latency; `qa_snapshot.jsonl` with options and answers |
| Per-photo and between-model statistics | `*/per_image.csv`, `*/answers.jsonl`, `*/frame_overlap.csv`, `*/image_manifest.csv` |
| Stricter overlap rules (same date, same tower) | `data/sft_train_qa.jsonl` and the photo names in `*/qa_snapshot.jsonl` |
| Crop analysis | the requested box of every episode (`answers.jsonl`), the human boxes (`gold_inputs/localization_mmad_qa_133.jsonl`) |
| Training description | everything in the training folder except checkpoints and weights (`train/train_metadata.json`, the step log if Drive has it), `data/build_summary.json`, `labels/*/annotations.json` |
| Reproducibility | `*/run_metadata.json` (model revision, adapter hashes, source commit, GPU), `logs/pip_freeze.txt`, `logs/nvidia_smi.txt`, `gold_inputs/refs_gold/` |
| Figures | `gold_inputs/preview/`: the 246 gold photos at 640 px with `index.csv` (name, original size) |
| Any other small classifier | `classifier/features.npz` and `feature_index.csv`: DINOv2 features of the training, gold and dev240 photos |
| Diagnosing a failed step | `logs/run_*.log` |

Not in the ZIP and not recoverable without a GPU: conditions or models that were not run, token probabilities
(the paper uses no score-based metric), and a second training seed.

## Result (2026-11-20, L4, source `e548776`)

The notebook ran in four sessions (three disconnects, each resumed from the saved calls). All six steps ended with
exit code 0, and the main rows did not change when the controls were added. Tables, per-image results and answers
are in `results/latest/agent_sft_gold_eval/`, which the paper branch `paper/explainability-motivation` adds together
with the paper section (archive `agent_sft_sft_v2_human_gold_results.zip`, sha256 `507ee879bbe1...`).

Primary scope, 226 photos (119 located):

| Row | Q1 F1 | Q2 macro-F1 | Joint | Missing cap, Q2 (of 23) | Position (of 119) |
|---|---|---|---|---|---|
| base, `direct_reference` | .415 | .499 | .283 | 0 | 0 |
| `sft_v1`, `direct_reference` | .909 | .848 | .885 | 11 | 50 |
| `sft_v1`, `adaptive_tools` | .909 | .838 | .881 | 12 | 7 |
| **`sft_v2_human`, `direct_reference` (primary)** | .883 | .849 | .867 | 10 | 60 |
| `sft_v2_human`, `adaptive_tools` | .882 | .824 | .867 | 9 | 73 |
| `sft_v2_human`, `fixed_evidence` | .793 | .873 | .721 | 16 | 54 |
| `sft_v2_human`, `matched_reask` | .888 | .849 | .872 | 10 | 59 |
| `dinov2_linear` | .844 | .729 | .845 | 3 | no position answer |
| constant "center" | | | | | 62 |

Dev240, `sft_v2_human`: joint .979 with `direct_reference` and .975 with `adaptive_tools`; `dinov2_linear` .954.

Verdicts under "What each outcome lets the paper say":

| Claim | Verdict |
|---|---|
| Fine-tuning carries over to gold | joint .867 against .283 before training; the same row is .979 on dev240 |
| The model locates defects | no: 60 hits against 62 for constant "center", CI of the difference [-.118, +.084] |
| Tool use helps | for position only: +.109 over `direct_reference` [+.025, +.193] and +.118 over `matched_reask` [+.034, +.193]; Q1, Q2 and joint do not differ |
| Choosing the crop helps | yes against `fixed_evidence`: position +.160 [+.059, +.261], joint +.146 [+.075, +.212] |
| The crop points at the defect | not met: the Q3 crop contains a box centre on 84 photos (of the 96 with a crop); the fixed box covers 90 of 119 |
| Human boxes and polymer photos help | position 60 against 50 (single call) and 73 against 7 (tool choice); joint 196 against 200 photos, CI includes zero |
| LVLM above the small classifier (Q1/Q2) | the row is above (.867 against .845), but the difference is five photos and its CI includes zero [-.022, +.066]; Q2 alone is above [+.009, +.097] |

Missing caps stay unsolved on gold: 10 of 23 named and none located, although all 25 are named on dev240. With the
fixed crop the model calls 53 of 101 normal photos anomalous. The between-model intervals are computed after the
run from `per_image.csv` (paired bootstrap, 2,000 resamples, seed 42) by `compute_sft_gold_differences.py` in the
paper folder of the paper branch.
