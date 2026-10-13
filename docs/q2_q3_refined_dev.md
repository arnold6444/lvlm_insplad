# Q2 rust / Q3 false-none refinement, dev240

Use `notebooks/lvlm_gold_v2_q2_q3_refined_dev_colab.ipynb`: fresh L4 runtime, existing GITHUB_TOKEN Secret enabled, Run all, authorize Drive. No annotation, box selection or answer entry. The default is InternVL on all 240 dev images; Qwen is optional. Existing QA, diffdate references, baseline predictions and ZIPs are read. Results go only to `gold_v2_q2_q3_refined_dev_v1`; the old 40-image folder, original baselines, gold and frozen settings stay unchanged. Results download as `dev_q2_q3_refined_results.zip`.

## What was actually diagnosed

The PR54 dev40 run had 232 calls, zero inference errors/invalid choices, and correctly parsed outputs. Q2 definition review changed three previously correct rust answers: two to normal, one to bird nest. Q3's five anomalous no-region answers comprise four negative presence checks that skipped position and one positive presence followed by no-region at position. These are confirmed routing/output observations; the model returned choices without reasons, so its internal cause is unknown.

Seven unique original target/reference pairs were inspected and their input bytes checked against the run manifest. Observations below are qualitative, not new spatial GT. They guided general rules only; no core IDs, filename rules, ground-truth labels, case-specific regions or per-case overrides are used by the method.

| Dev core | Observed output/failure | Visual/prompt hypothesis |
|---|---|---|
| dev-fef1268642e6aa79 | rust→bird nest in Q2 | dark rods/rough metal beside background twigs; surface corrosion vs external twigs may be confused |
| dev-65c79ac3c603fcff | rust→normal in Q2 | dark mottled metal link/rod without dominant bright-orange colour; white coatings and different illumination complicate comparison |
| dev-d413e05b608c5542 | rust→normal in Q2; Q3 negative presence | dark rough link/hardware, similar overall shape to normal reference but different pose/light |
| dev-676efe18c902578c | Q3 presence positive, position no-region | metal caps remain around an interrupted glass-disc/skirt structure; localization task may lose missing-material evidence |
| dev-65b1682ed9bac3b3 | Q3 negative presence | dense repeated glass structure and different reference viewpoint; exact missing region is not established by this audit |
| dev-30f82a43a929b5ce | Q3 negative presence | extensive dark textured rod/metal rather than a compact orange patch; legacy Q2 also misclassifies it as missing-cap |
| dev-fcdd0fac74af5c64 | Q3 negative presence, Q2 correctly bird nest | thin intertwined twigs around hardware blend into vegetation; two questions can disagree even with correct Q2 |

The old rust definition already includes dark-brown corrosion, and missing-component definition already includes discs. The refinement clarifies those cues and competing normal/background explanations; it does not claim the old definitions omitted them entirely. The new Q2 prompts remove previous-answer/review framing; a fresh-old-definition ablation preserves the same introduction, definitions, options and output request to compare framing with revised definitions. Prior-answer framing causality is not established by the diagnosis.

All seven targets are nearly square; applying the actual adapter's two-image budget (max_total_tiles=6 → max_num=3/image) yields one 448×448 tile each. Some source images exceed 1000px, so fine-detail loss is possible. This experiment retains the same image processing and decoding across policies; it tests cues/routing first and does not establish or fix resolution as a causal explanation. References are different dates and can differ in pose/light; this is also held fixed.

## Conditions

| Policy | Q2 | Q3 |
|---|---|---|
| baseline | cached | fresh legacy direct grid question |
| previous_agent | exact legacy own-answer define | exact legacy presence→optional position |
| fresh_define | exact legacy definitions, no previous answer/review | legacy direct |
| q2_cues | fresh balanced revised definitions | legacy direct |
| q2_cues_q3_position | revised Q2 | revised grid question on every image |
| q2_cues_q3_review | revised Q2 | revised presence and position on every image; review disagreements |
| matched_reask | legacy own-answer reask | legacy direct repeated twice/three times, matching refined Q3 budget |

Q1 is never inferred and is asserted identical to its cached stage1 answer. Q2 never reads Q1/Q3 outputs. Q3 never reads Q1/Q2 outputs or GT. Every tool uses reference first and the original full target second, fresh single-turn calls, the same adapter/tile budget and greedy decoding, no quantization. The controller uses valid option semantics rather than fixed letters, as option order is shuffled.

Revised definitions cover dark/black mottled or flaky corrosion (shadows/smooth paint alone insufficient), missing glass/discs even if caps remain (occlusion alone insufficient), attached thin nesting material (background branches/cable strands insufficient), and normal surfaces not needing perfect shine/uniform colour. These general cues are not trained or calibrated numeric thresholds.

Q3 always runs both its own presence and its own position task. A valid yes+region or no+none pair uses position. A no+region or yes+none pair gets one own-Q3 review with both earlier answers and the images; neither earlier answer is forced. An invalid review retains valid own position, then own direct if position is invalid. Invalid presence does not veto a valid position; invalid tool counts remain reported. Exceptions exclude affected policies and exit nonzero, even when another answer could be used. No unconditional preservation of a baseline rust/region answer: such a rule could inflate false positives.

Main comparisons include baseline vs each policy, previous_agent vs refined review, fresh_define vs revised Q2, revised Q2 vs always-position, always-position vs review, and same-call reask vs review. Changes in visual cues and presence semantics are combined in the Q3 refinement; these ablations do not isolate every phrase. Same-model calls are correlated, not independent expert models. Actual benefit remains unverified until Colab execution.

## Larger-dev evaluation and preservation

Default dev240 = 125 normal / 115 anomalous. `design_dev40` is fixed to the original diagnostic subset at seed42 (24 normal /16 anomalous), even if the additional sampling seed changes. `additional_dev` excludes all design cores: 200 images, 101 normal /99 anomalous. `all_dev`, `design_dev40`, and `additional_dev` get separate metrics, per-asset/per-defect support, and image-paired 2000-resample CIs. Inspect additional_dev first, including rust recall, normal false-region and anomalous false-none. The additional group was excluded from this seven-case diagnosis, but is existing development data with prior baseline experiments; it is not a new formal test holdout. No selection uses these scores automatically.

Q3 retains the PR54 partial-label contract: nine original-frame regions+none, shuffled without GT; normals get a none answer, anomalous spatial answers remain null. Metrics only assess none-versus-region against image-level GT. Wrong grid cells on anomalous images can still be presence-correct. Q3 location accuracy remains null, with zero anomalous spatial support. Joint correctness is Q1+Q2+Q3-presence only, not original gold localization. Always-none/region controls and invalids are retained. Per-asset single-class metrics must be read with support.

Cost: 10–13 unique new tool calls/image → 2400–3120/model. Cache and shared outputs avoid rerunning a tool per policy. Matched reask and full refined review have equal image-level logical call counts; report these separately from unique calls. No Q1 calls. Runtime has not been measured for this larger experiment.

Twenty-call checkpoints bind prompts, Q1/Q2 source answers, QA, references, model specs, current source ZIP bytes, and the refined/common runner code hashes. Changed inputs/code require a new output folder. Successes are reused; failures/blank responses retry. If a previously failed call changes the route, newly required review/control calls also run. Failed policies are excluded from all cohort comparisons to preserve equal full coverage. Diagnostics/answers remain available; no selection/frozen config or gold run is written. The legacy runner's default protocol/output identity is preserved.
