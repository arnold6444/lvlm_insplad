# Question-independent verification agent (gold v2)

Research question: does re-checking each question on its own improve accuracy, and how many cross-question
contradictions remain when it does? Runner: `scripts/run_gold_v2_verify_agent.py` (protocol
`gold_v2_verify_agent_independent`, prompt version 2). It replaces the PR #49 design, which chose images by their
Q1-Q2 contradiction and rewrote Q1/Q2/Q3 from one Q2 re-check.

## Independence contract

- A re-check of question Qk sees the images ([retrieved normal reference, target]), Qk's text and options, and Qk's
  own stage-1 answer. Nothing from any other question.
- Whether Qk is re-checked is decided from Qk's own answers only (`all`, or `disagreement` = Qk stage-1 answer vs Qk
  image-only answer).
- A re-check writes only Qk's answer. Valid answer → final; invalid → stage-1 answer kept.
- Every call is a fresh single-turn chat (`infer_hf` keeps no history).
- Q1-Q2 / Q1-Q3 contradictions are computed after all final answers are stored and only reported.
- `tests/test_gold_v2_verify_agent.py::test_changing_other_questions_answers_does_not_touch_this_question` rewrites
  every other question's stage-1 and image-only answers and checks that Qk's routing, prompts, images, re-check
  answers and final answers do not change (Q1, Q2 and Q3).

## Variants

`<tool>@<route>`, tools `reask` (control) / `zoom` (central 60% crop ×2, not applied to Q3) / `define` (definitions
of the question's own options) / `zoom_define`; routes `all` / `disagreement`. A re-check is made once per question
and tool and shared by the routes; a route's cost counts only the questions it re-checks. The removed
`rule_q1_first` baseline rewrote Q2/Q3 from Q1 and is not part of this protocol.

## Dev, then gold once

1. Dev (240 images, Q1/Q2 only, InternVL + Qwen): all variants. `selection.csv` ranks them by the pre-set criterion
   — mean over models of the change in (Q1 F1 + Q2 macro-F1) / 2 vs stage 1, ties to fewer calls; contradiction is
   shown but not scored.
2. `make_frozen_config` writes `frozen_config.json` (one variant, stage-1 condition, rule, prompt version); it is
   published with the dev results.
3. Gold (`--qa-dir`) refuses to run without `--frozen-config`, refuses `--tools/--routes`, checks the config against
   the run, and refuses a folder that was already run with a different config.

Q3 has no dev labels (dev `build_summary.json`: "No Q3 (no location labels)"); gold Q3 uses the frozen variant and is
reported as auxiliary next to the constant baselines (always "no region" 0.524, always "center" 0.476).

## Outputs (`results/latest/gold_v2_verify_agent_{dev,gold}/`)

`leaderboard.csv` (Q1 F1, Q2 macro-F1, per-question accuracy, Q3 on gold, contradiction and new/resolved counts,
joint, FDR, re-checks, calls per question, latency), `per_question.csv` (re-checked / changed / changed to correct /
to wrong), `metrics_by_asset.csv`, `answers.csv` (stage-1 and final answer of every question), `significance.csv`,
`selection.csv` (dev), `route_stats.json`, `run_status.json`, `run_metadata.json`, `frozen_config.json` (dev).
