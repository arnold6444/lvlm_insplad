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
