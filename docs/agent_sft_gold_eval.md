# Gold scoring of the fine-tuned agent (protocol fixed 2026-11-17, not run yet)

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
  What remains is scoring, about 5 to 6 hours on an L4.

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
- **Conditions.** `direct_reference` (one call) and `adaptive_tools` (up to three calls). The fixed-crop and
  matched re-ask controls already ran on dev240 and are left out to save calls.
- **Models, in run order.** The order is by importance, so an interrupted run keeps the main result.

| Order | Model | Why |
|---|---|---|
| 1 | LoRA `sft_v2_human` (human boxes, polymer added) | **primary** |
| 2 | Qwen3.5-4B before training | same prompts, no training |
| 3 | dev240, LoRA `sft_v2_human`, the same two conditions, in a new folder | the dev row the paper's dev table lacks |
| 4 | LoRA `sft_v1` (automatic boxes), only if its adapter is on Drive | what the human boxes and polymer photos add |

The interrupted four-condition dev240 run cannot be resumed by this runner (a run is bound to its source), so it
is left untouched and its saved answers go into the result ZIP as `dev_lora_earlier_four_condition_run/`.

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
- **Intervals.** Paired bootstrap, 2,000 resamples, seed 42: `adaptive_tools` minus `direct_reference`, and each
  condition's position hit rate minus the constant "center" answer.

## What each outcome lets the paper say

Written before the run. No threshold is set for "keeps its score"; the gold numbers are stated next to dev240's.

| Claim | Allowed only if (primary scope, `sft_v2_human`, `direct_reference` unless noted) |
|---|---|
| Fine-tuning carries over to gold | report joint accuracy and Q2 macro-F1 as they are, with the base row and the dev240 row |
| The model locates defects | position hit rate minus constant "center" has a 95% CI above zero |
| Tool use helps | `adaptive_tools` minus `direct_reference` has a 95% CI above zero on that metric |
| The crop points at the defect | the Q3 crop covers a box centre more often than the fixed center-60% box (90 of 119) |
| Human boxes and polymer photos help | stated as a difference between two runs that changed three things at once, never as one cause |

If position does not beat "center", the paper says location is unsolved. If the gold scores are far below dev240,
the paper says the dev240 scores reflected photos close to the training set.

## Limits

- One model, one training run per adapter, greedy decoding.
- Gold was used before by other methods, and its photos come from the same site and campaign as the training
  photos. Frame-disjoint removes shared drone frames, not shared dates or towers.
- The prompts and the retrieved reference differ from the paper's gold tables, so these rows are comparable with
  each other and with dev240, not with the baseline tables.
- No small supervised classifier is trained on the same labels, so nothing here shows that an LVLM is the better
  use of the labels.
- Estimated time on an L4 (from the first run's speed, not measured): gold about 4 h in total, dev240 1.5 h,
  and 6-7 h more only if training has to run.

## Safeguards for a single execution

- The first cell reads the GPU with `nvidia-smi` before Drive and pip, and stops with the attached GPU's name.
  The notebook metadata asks for an L4.
- The location set is embedded in the notebook and checked by SHA-256.
- Right after the training data is built, the gold runner runs with `--inputs-only`: gold QA, location set,
  references, ZIP members and the frame overlap are all checked before any training time is spent.
- Every step resumes from Drive: training checkpoints every 50 steps, saved calls are never repeated.
- `RUN_GOLD_BASE`, `RUN_DEV240` and `RUN_GOLD_SFT_V1` in the settings cell skip a step; a skipped row is absent.

## Outputs

`MyDrive/lvlm_insplad_results/agent_sft_gold_sft_v2_human/agent_sft_gold_eval_*/` per model: `leaderboard.csv`
(one row per scope and condition), `metrics_by_asset.csv`, `q2_by_defect.csv`, `position_by_defect.csv`,
`per_image.csv`, `significance.csv`, `frame_overlap.csv`, `run_metadata.json`, `run_status.json`, saved calls.
Everything is also in `agent_sft_sft_v2_human_gold_results.zip` on Drive.
