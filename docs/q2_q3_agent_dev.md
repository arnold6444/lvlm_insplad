# Q1 fixed, Q2/Q3 independent dev experiment

Open `notebooks/lvlm_gold_v2_q2_q3_agent_dev_colab.ipynb` on merged main, use a fresh L4 runtime, enable the existing `GITHUB_TOKEN` Secret, then Run all. Drive authorization is required; annotations, boxes and manual answer entry are not. Existing dev QA, diffdate reference CSV, baseline predictions and source ZIPs are read. The default is InternVL on the same deterministic dev40 as the prompt diagnosis (seed42); Qwen is optional, with twice the total calls.

## Conditions and independence

| Policy | Q1 | Q2 | Q3 |
|---|---|---|---|
| baseline | cached | cached | fresh direct question |
| q2_define | cached | own-answer definition review | direct question |
| q2_define_q3_independent | cached | own-answer definition review | separate presence task, then optional full-frame position task |
| matched_reask | cached | own-answer reask | repeated direct question, same image-level call budget as the agent |

Q1 answers are asserted identical across policies; Q1 has zero inference calls. Q2 uses the existing define/reask prompts unchanged and only its own first answer. Q3 presence/routing/position never sees Q1 or Q2 answers, defect labels, or evaluation GT. A valid tool answer replaces that question's own answer; invalid Q2 retains cached Q2, invalid Q3 tool retains its own direct Q3. Invalid direct Q3 remains invalid. Every call uses normal-reference first and original target second with the existing greedy adapters and no quantization. This is a tool pipeline within the same LVLM; it introduces no detector or vision expert.

The Q3 presence task does not consult its own direct answer, avoiding prior-answer anchoring. Presence says none -> none. Presence says region -> ask for the position on the original full-frame grid. The position answer can still be none. A matched control repeats direct Q3 once (or twice when the agent requested position). Greedy repeat calls may give identical answers; this control measures extra calls with no new task instructions. Definition prompts also include review and the previous own Q2 answer, so this experiment does not isolate the causal effect of definitions alone.

## Partial Q3 evaluation contract

Dev provides image-level normal/anomalous labels, not spatial GT. Generate nine original-frame grid choices plus no-region, deterministically shuffled from core ID and protocol version without GT. Normal rows receive the no-region answer; anomalous rows have `answer=null`. This ten-choice auxiliary task differs from the original gold four-choice task and its center-biased labels. Neither gold answers nor gold images are used.

Score only **none-versus-region against image-level labels**: normal no-region accuracy / false-region rate, anomalous region recall / false-none rate, balanced accuracy, invalid rate, and Q1+Q2+Q3-presence joint correctness. All location accuracy fields remain null; anomalous location support is zero. Even a wrong grid cell on an anomalous image counts as region-present, so this score must not be called localization accuracy. Image-level anomalous GT is a proxy for visible localized defects; actual visibility/localizability is not verified. Normal none scoring assumes the dataset normal labels are correct.

Always-none and always-region controls each have balanced accuracy 0.5 when both classes exist. Report both normal and anomalous supports, invalids and raw accuracy. Image-level paired bootstrap CIs compare each policy to baseline and the combined agent to matched reask. Dev40 CIs are exploratory; neither these CIs nor the presence-only joint metric establishes gold localization gains.

## Cost, preservation and resume

InternVL default: 200–280 unique new Q2/Q3 calls, zero Q1 calls. Two Q2 tools, one Q3 direct, one presence, one matched repeat per image; add position and a second matched repeat when presence says region. Shared outputs are reused across policies; per-policy call counts are logical budgets, not their sum. Actual GPU time/effect must be measured in Colab.

Only the dedicated `gold_v2_q2_q3_independent_dev*` output directory is written. Original QA, baselines, gold and frozen settings are unchanged. Store raw responses, prompts, choices, tool failures, image SHA256, model specs, commit, software versions, answers, per-asset/per-defect metrics, controls and significance. Save every 20 used calls. Same-folder resume binds QA, prompts, own stage1 answers, references, model specs and original ZIP image bytes; changed inputs require a new folder. Successful nonblank predictions are reused, failed/blank calls are retried. Changing a previously failed presence decision may add needed position/control calls.

Baseline missing/error/invalid answers block preflight. Failed inference excludes every dependent policy from leaderboard and significance, preserves raw diagnostics, and exits nonzero so the notebook stops. Non-error invalid responses remain visible as invalids and follow the stated fallback rules. The runner never writes a selection or frozen configuration and never runs gold. The final notebook cell downloads `dev_q2_q3_agent_results.zip`; Drive outputs remain saved. After reviewing results, any future configuration/gold experiment is a separate decision.
