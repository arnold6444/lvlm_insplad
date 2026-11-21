# Plan — final handoff (2026-10-24)

## Goal
Finish the InsPLAD-QA paper. All experiments are done and the scope is frozen (user decision, 2026-10-23).

## Decisions
- No new experiment unless it changes a paper table or sentence, and only when the user asks for it.
- The missing-component branch is closed; its results are a limitation and an appendix table in the paper.
- Gold v2 (246 images / 534 questions) stays FROZEN. Gold is never used to choose a setting.
- Full Track A/B inference (1,378 rows or more) runs only on an explicit request.
- Commits, pushes and result publication happen only on the user's request.

- **2026-11-17 (user): the scope is reopened for one addition.** The LoRA fine-tuning follow-up goes into this paper as a
  "few hundred labelled images" section. Everything else stays frozen. Code and notebooks for it are on `main`
  (`docs/agent_sft_lora.md`); the paper text is on `paper/explainability-motivation`, committed locally on 2026-11-21, not pushed.

## Fine-tuning section (written and committed 2026-11-21, not pushed)
- Status 2026-11-21: items 1-4 below are done. The section now has the second run on dev240, the gold protocol,
  table `tab:sftgold` (`tables/sft_gold_rows.tex`), the classifier comparison, position, crops and the limits; the
  abstract, introduction (fourth contribution), Section 6 item, discussion item, conclusion and reproducibility
  appendix were revised in both languages. Result tables: `results/latest/agent_sft_gold_eval/` (manifest with the
  ZIP hash); model differences: `compute_sft_gold_differences.py` -> `model_differences.csv`.
  Checked: 12 table rows and 26 text items recomputed from the result CSVs (all equal), static checks pass, both
  Word files rebuilt (14 tables). Not done: LaTeX compile, push, PR, merge of `feat/sft-gold-eval` (the protocol
  document the paper cites lives on that branch).
- Supported: fine-tuning raises presence and type on frame-disjoint gold (64 -> 196 of 226); not clearly above a
  small classifier on the same labels (191; Q2 alone is above); tool choice locates more often than the single
  call and both controls (73 vs 60, 59, 54 of 119). Not supported: position above constant "center" (62),
  tools helping presence or type, labelled examples solving missing caps (10 of 23 on gold).
- Earlier steps, kept for the record. Done: draft section `sec:sft` and table `tab:sft` in `main_ko.tex` / `main_en.tex` with the first run (`sft_v1`, 600 photos,
  automatic boxes) on dev240 only. Rows: `reports/paper_draft_2026-10-23/tables/sft_dev_rows.tex`. Static checks pass.
- Not done, in order:
  1. Second run (`sft_v2_human`, 660 photos, human boxes) and gold scoring, in ONE Colab execution (user, 2026-11-17:
     Colab can be run only once more). Prepared, not run: branch `feat/sft-gold-eval` (worktree
     `C:/Users/user/Desktop/lvlm_insplad_gold_eval`, commits 19c651d + 7d6ff47, pushed to origin, no PR), notebook
     `notebooks/lvlm_agent_sft_lora_human_boxes_gold_colab.ipynb`, protocol `docs/agent_sft_gold_eval.md`.
     Order: train -> gold (second run) -> gold (base) -> dev240 (second run, two conditions) -> gold (first run).
     Drive (read 2026-11-19): the second run is already trained (2026-11-17, L4, 599 steps, val loss .779 -> .143);
     its dev240 run stopped midway; no gold scoring yet. The notebook was rebuilt to skip training and to write
     dev240 to a new folder (commits 6e8ebfa + 6457daf, pushed to origin/feat/sft-gold-eval on 2026-11-19, no PR).
     Same day, on the user's instruction that the result ZIP must lack nothing: gold controls for the second model
     (fixed crop, matched re-ask), a ZIP rewritten after every step, a log on Drive, small gold photos for figures,
     and by user decision a small classifier on the same 660 labels (DINOv2 features + linear).
     Commits 515c8d0, e548776, 3878927 (pushed to origin/feat/sft-gold-eval on 2026-11-19, no PR); the 6457daf link is the old version.
     Rehearsed end to end on the real 246 gold photos without a model. What is left is scoring, about 6-7 h on an L4
     (estimate). When the ZIP arrives, the limitation "no small classifier" in the draft section must be rewritten.
     2026-11-21: the run finished (L4, source e548776, four sessions, all six steps exit 0) and the ZIP was checked
     (`C:/Users/user/Downloads/agent_sft_sft_v2_human_gold_results.zip`, not yet in the repo). Per-image recount
     matches the leaderboards. Frame-disjoint gold, 226 photos: Q1+Q2 both correct 64 (base) -> 196 (second run,
     single call; first run 200; small classifier 191); dev240 second run 235/240. Position, 119 located: single
     call 60, always-center 62, agent 73 (above the single call and both controls, CI excludes 0; not above
     always-center, CI includes 0). Missing cap stays unsolved on gold (Q2 10 of 23). Not yet in the paper.
  2. Gold scope: 20 of the 246 gold photos share a drone frame with a second-run training photo (6 normal,
     14 defective), so the primary scope is 226 photos (119 of the 133 boxed). Constant "center" scores 62 of 119.
     Checked locally with the real labels, photo names, location set and training list; no model call.
  3. Abstract, introduction, discussion and conclusion: revise only after the gold result.
  4. Copy the run summaries into `results/latest/`, and rebuild the Word files.
- (2026-11-17 note, superseded by the status above: "Not claimed: that tools help, that location is learned, or
  that an LVLM beats a small classifier (never compared).")

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
The gold result is in the paper (2026-11-21, on the user's request "논문에 반영") and committed locally the same day (four commits, 6458e26..a4bc530 plus this note), not pushed. Next, when the user
asks: read the new section and confirm its wording, commit and open PRs (paper branch, and `feat/sft-gold-eval`
for the code and protocol), compile on Overleaf, fix title and authors.
