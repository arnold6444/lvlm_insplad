# Plan — final handoff (2026-10-24)

## Goal
Finish the InsPLAD-QA paper. All experiments are done and the scope is frozen (user decision, 2026-10-23).

## Decisions
- No new experiment unless it changes a paper table or sentence, and only when the user asks for it.
- The missing-component branch is closed; its results are a limitation and an appendix table in the paper.
- Gold v2 (246 images / 534 questions) stays FROZEN. Gold is never used to choose a setting.
- Full Track A/B inference (1,378 rows or more) runs only on an explicit request.
- Commits, pushes and result publication happen only on the user's request.

## Completed
- Benchmark, gold v2 and dev set; baselines; references; vision-expert cues; verification agent on dev and gold.
- Matched-call control on gold (PR71/72): the agent beats an equal-call re-ask, +.164 Q2 macro-F1 and +.203 presence BAcc, and both CIs exclude zero.
- MiniCPM-V-4.5 greedy re-run (PR73/74), with the required anomaly/structured metrics re-aggregated on CPU (PR77).
- Paper draft EN/KO with Word versions (PR72–76). The numbers were checked independently (`reports/paper_draft_2026-10-23/VERIFICATION_2026-10-24.md`).
- Review fixes (PR77):
  - notebooks accept a run's files only when the exit code is 0 and the invocation ID matches;
  - the runners refuse to load models below 20 GiB VRAM;
  - manifest checksums cover the committed bytes.
- Final cleanup (2026-10-24):
  - Removed notebooks/scripts/tests that no paper result depends on: v1 gold, early smoke runs, AgentIAD-lite, and unreported heatmap/part-crop diagnostics.
  - Added the per-experiment `analysis.md` summaries that the docs link to.
  - Rewrote README/GOAL/plan and added `docs/README.md`.
  - Code and raw logs of the appendix experiments that were never committed are archived locally (`reports/_archive/final_cleanup_2026-10-24/`, gitignored).

## Where things are
- Paper: `reports/paper_draft_2026-10-23/` (`main_en.tex`, `main_ko.tex`, `tables/`, `refs.bib`, `InsPLAD_QA_paper_{en,ko}.docx`).
- Results: `reports/paper_tables/latest_results.md` and `results/latest/`.
- Notebook map by paper section: `README.md`.

## Remaining
- Compile in Overleaf (`main_en.tex`, `main_ko.tex`; pdfLaTeX, kotex for Korean). LaTeX was only checked statically, never compiled.
- Fix the title, authors and affiliations.
- Confirm the InsPLAD volume/pages in `refs.bib` (the DOI is confirmed).

## Next action
Upload the paper folder to Overleaf and compile both versions.

## Requirement delta — tool-choice pilot (2026-11-04)

- User authorized a bounded follow-up and GitHub upload after local verification. This does not reopen training or the frozen paper experiments.
- Goal (updated at user's request): one self-contained Colab Run all for dev240, with no separate dev40/manual gate and no dependency on GitHub publication. Use one >=20 GiB GPU. No local HF inference, SFT, GRPO, web search, gold tuning, automatic publication of results, or fabricated expected scores.
- Preserve existing QA, predictions, paper drafts and the older dev/gold protocols. Work is isolated on `codex/agent-tool-pilot`.
- Flow: each Q1/Q2/Q3 independently uses original image, optional retrieved different-date normal and at most one normalized original-frame crop. The model emits strict JSON for reference/crop/finish; Python executes tools and feeds results back. Maximum three LVLM calls per question.
- Controls: direct normal-reference answer, fixed reference+central crop, adaptive tools, and reference-only re-ask matching the adaptive call count. These are new pilot controls, not a rerun of the older verification pipeline. Keep image/tile and prompt confounds explicit.
- Dev spatial GT is absent. Report Q1/Q2 metrics, FDR/consistency and Q3 presence metrics. True Q3 location and Q1+Q2+Q3-location joint stay null; never substitute presence joint for location joint.
- Steps: implement runner and strict resume/input identity; add CPU contract tests; create SHA-pinned Colab with Drive input discovery, GPU guard and result ZIP; inspect diff, test, publish branch/PR, verify remote files.
- Done when for the revised single-file delivery: CPU tests cover adaptive execution, invalid/tool failures, budget, label isolation, resume, extraction safety and missing spatial GT; notebook cells compile, source pin and embedded hashes validate, and extracted source imports without GitHub or `.git`. Requested remote publication remains separately blocked; neural effectiveness remains unrun until a Colab GPU run returns evidence.
- Checkpoint: runner, notebook builder and protocol implemented. Targeted CPU verification: 36 passed (20 new contracts plus 16 related existing tests), `git diff --check` clean. Critical self-review added a CLI GPU-memory guard, model-load abort, resolved-model revision guard, same-date/content checks, per-call identity and invocation matching. No synthetic scores are published. Remaining: generate notebook against committed source, verify its cells/pin, publish and check remote artifacts.
- Notebook checkpoint: generated against immutable source `150aa27f8840870729b9f842ac62c5b79d1e35c6`; four code cells compile and contain no execution outputs. Final 20 new tests pass after compatibility/provenance fixes. Preparation evidence is in `reports/agent_tool_pilot_preparation_2026-11-04.md`. Remaining: remote publication verification and user-run GPU pilot; no effectiveness result is claimed.
- Publication blocked: automatic approval review rejected `git push -u origin codex/agent-tool-pilot` because approval is required while the current approval policy is Never. No remote branch/PR was created (`git ls-remote --heads origin codex/agent-tool-pilot` empty). Local commits are ready. After an authorized manual push, the immutable notebook/source URLs become available; GPU execution remains pending. Generated test scratch is untracked and excluded from commits; its cleanup was also denied by automatic review.
- One-run requirement delta: replace remote checkout with a source archive embedded from a committed snapshot, validate archive/member hashes and report that source SHA in runtime receipts. Default dev240, four controls and result download in one pass. Preserve diagnostic ZIP export even on inference failure; never label it success. This is local artifact delivery, not a bypass or retry of the denied GitHub write.
- Revised delivery verified: current source pin `c9b3f10a996f52f3bec20904da006533de05eb23`, 105 embedded source/config files, four compiled code cells, no execution output. Standalone extracted runner returns the correct validated source SHA without `.git`. Current targeted tests: 22 passed. User action: upload notebook to Colab, select >=20 GiB GPU, authorize Drive and Run all once. No dev40/manual git step is required.

## Requirement delta — LoRA SFT of the tool policy (2026-11-09)

- Pilot result (L4, InternVL3.5-8B, PR #84 notebook): complete, 0 row errors, 4,341 calls. Training-free tool choice did not help. Q2 macro-F1 was .431 (direct_reference) vs .328 (adaptive_tools, 44% invalid, mostly `"A."`). Q3 presence was about .5 for every condition.
- User decisions: Qwen3.5-4B BF16 LoRA on L4; automatic DINOv2 patch-difference crop boxes; template rationale. Implemented as JSON-only actions because the strict eval parser rejects text. No GRPO, gold or new holdout in this step.
- Data: supervised train minus dev240 images, fingerprints and same drone photos (date + DJI frame), minus unsup test fingerprints, polymer corrosão and fixed refs. 60 per class. A date-disjoint split is impossible because dev240 covers all 35 train dates. Training defect photos share frames with unsup test, so gold scoring needs a frame-overlap filter first.
- Done when: CPU tests show trajectories replay through the eval runner with identical prompts, the label mask covers only action + `<|im_end|>`, the adapter is bound into the eval signature, and notebook cells compile from a committed source pin. GPU training/evaluation is run by the user.
