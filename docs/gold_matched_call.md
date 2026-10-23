# Matched-call control on gold (structure vs. more calls)

## Question

The gold verification agent beats the direct question (Q2 macro-F1 .522 → .640, Q3 presence BAcc .551 → .828), but it also spends more LVLM calls. This control asks whether the gain comes from the agent's **structure** (option definitions for Q2, presence-then-position staging for Q3) or simply from **asking again**.

## Run

Open `notebooks/lvlm_gold_v2_matched_call_colab.ipynb` in a fresh **L4/A100** Colab runtime → **Runtime → Run all** → authorize Drive → return `gold_matched_call_results.zip`. No token, labels or manual steps. The notebook clones public `main` and refuses to run unless every code file it imports has the git blob ID the notebook was built with.

It reuses the completed PR56/57 gold run in `MyDrive/lvlm_insplad_results/gold_v2_q23_candidate_v2/` (and its normal references) when present. A saved step is reused only when its prompt and image bytes are identical, so the three published policies stay exactly as reported. A CPU check against the real exported run matched all 1,606 saved steps; **622 new calls** remain (Q2 246, Q3 376). Output: `MyDrive/lvlm_insplad_results/gold_v2_matched_call_v1/`; earlier folders are read only. Re-running the inference cell resumes the same run.

## Policies (locked in `configs/gold_v2_matched_call_v1.json`)

| Policy | Q2 | Q3 auxiliary |
|---|---|---|
| baseline | direct (1 call) | direct (1 call) |
| previous_agent | direct + own answer + definitions (2 calls) | presence, then position if present (1–2 calls) |
| candidate | revised definitions, no prior answer (1 call) | same as previous_agent |
| **matched_reask** | direct + own-answer re-check, no definitions (2 calls) | direct + k sequential own-answer re-checks, k = the staged Q3's calls for that image |

Q1 is the fixed direct answer in every policy (zero Q1 calls). `call_budget.csv` shows that `matched_reask` and `previous_agent` spend the same Q2 and Q3 calls. The dev runner's Q3 control repeated the identical greedy direct prompt, which is degenerate. Here each re-check shows the question's own previous answer, like the Q2 re-check. No other question's answer is ever shown.

Primary comparison: `previous_agent − matched_reask` (paired bootstrap, 2,000 resamples) on Q2 macro-F1, joint accuracy and Q3 presence BAcc / normal false-region / anomaly false-none. It also reports baseline vs. matched_reask and candidate vs. matched_reask. This is an analysis-only control: no setting is chosen or changed from gold scores. Q3 is presence only and `q3_location_accuracy` stays null.

## Gates

The run is complete only when all four policies have full coverage with zero errors and zero invalid outputs. Invalid or failed policies are excluded from tables and comparisons, and only failed or invalid calls are retried. The run stops on changed inputs, lock, code, images or reused-run image bytes, so use a new folder in that case. Outputs follow the gold candidate runner (`leaderboard.csv`, `answers.csv`, `metrics_by_asset.csv`, `q2_by_defect.csv`, `significance.csv`, raw `steps.jsonl`, metadata/status) plus `call_budget.csv`.
