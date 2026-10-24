# Experiment protocols

Each file describes one Colab experiment: its inputs, fixed settings, pass/failure gates and outputs. All experiments are finished; the results are in [`reports/paper_tables/latest_results.md`](../reports/paper_tables/latest_results.md) and the paper.

## Vision-expert cues (paper §4)
- [vision_expert_smoke.md](vision_expert_smoke.md): detector screening smoke; its artifacts feed the gold v2 selection notebook

## Verification agent (paper §5)
- [verify_agent_independent.md](verify_agent_independent.md): the question-independent verification agent
- [q2_q3_agent_dev.md](q2_q3_agent_dev.md): Q1 fixed, Q2/Q3 re-checked separately on dev
- [q2_q3_refined_dev.md](q2_q3_refined_dev.md): rust and Q3 false-none refinement on dev240 (the candidate)
- [q23_candidate_gold.md](q23_candidate_gold.md): the single gold run of the locked policies
- [gold_matched_call.md](gold_matched_call.md): equal-call re-ask control on gold
- [gold_minicpm_greedy.md](gold_minicpm_greedy.md): greedy re-run of the MiniCPM-V-4.5 gold conditions

## Missing-cap follow-ups, dev only (paper §6 and appendix)
- [q23_zoom_prompt_dev.md](q23_zoom_prompt_dev.md): target zoom × short prompt
- [q3_multiregion.md](q3_multiregion.md): separate nine-cell multi-region Q3
- [dev_missing_location_annotation.md](dev_missing_location_annotation.md): location boxes for 25 dev missing caps
- [q2_conflict_evidence_dev.md](q2_conflict_evidence_dev.md): per-candidate Q2 checks with an equal-call control
- [q3_gt_crop_diagnostic_dev.md](q3_gt_crop_diagnostic_dev.md): crops of the annotated location
- [q3_missing_occlusion_dev.md](q3_missing_occlusion_dev.md): enlarged location candidates
- [missing_component_probe.md](missing_component_probe.md): asset-specific supervised missing-cap classifier
- [selective_missing_expert_agent_dev.md](selective_missing_expert_agent_dev.md): normal reference + frozen classifier + LVLM
- [missing_presence_independent_dev.md](missing_presence_independent_dev.md): presence asked without earlier answers
- [missing_expert_direct_dev.md](missing_expert_direct_dev.md): classifier and location candidates adopted directly
- [missing_ig_location_dev.md](missing_ig_location_dev.md): integrated-gradients location tool
- [missing_ig_numerics.md](missing_ig_numerics.md): numerical diagnostic of the integrated-gradients run
