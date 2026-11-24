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
  (`docs/agent_sft_lora.md`); the paper text is on `paper/explainability-motivation`, committed and pushed on 2026-11-21 (PR91).

## Fine-tuning section (written 2026-11-21, PR91)
- Status 2026-11-21: items 1-4 below are done. The section now has the second run on dev240, the gold protocol,
  table `tab:sftgold` (`tables/sft_gold_rows.tex`), the classifier comparison, position, crops and the limits; the
  abstract, introduction (fourth contribution), Section 6 item, discussion item, conclusion and reproducibility
  appendix were revised in both languages. Result tables: `results/latest/agent_sft_gold_eval/` (manifest with the
  ZIP hash); model differences: `compute_sft_gold_differences.py` -> `model_differences.csv`.
  Checked: 12 table rows and 26 text items recomputed from the result CSVs (all equal), static checks pass, both
  Word files rebuilt (14 tables). Not done: LaTeX compile. The runner and the protocol
  document the paper cites are on main since PR92 (merged 2026-11-22).
- 2026-11-24, after two points raised by a reader and confirmed by the user (branch `paper/sft-review-fixes`, from main
  f932afd): the dev240 table and its details moved to an appendix, because dev240 comes from the training pool; the
  untrained model was re-scored without format-only errors (`scripts/score_agent_sft_gold_lenient.py`,
  `results/latest/agent_sft_gold_eval/gold_base_lenient/`: 64 -> 98 of 226 photos both correct, strict row unchanged);
  and the paper now says that all 226 primary-scope gold photos share a capture date with a training photo (the
  training photos cover all 35 dates). Abstract and conclusion were reworded accordingly. Word files rebuilt.
- Supported: fine-tuning raises presence and type on frame-disjoint gold (64 -> 196 of 226; 98 for the untrained model without format errors); not clearly above a
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
- Fix the title, authors and affiliations. The title was reworded on 2026-11-24 after an external review ("Answer
  Consistency and Re-Verification"); it is a proposal until the authors confirm it. The reproducibility section now
  links the repository, which must be replaced or removed for a blind submission (the author line says anonymous).
- Confirm the InsPLAD volume/pages in `refs.bib` (the DOI is confirmed).

## Next action
The gold result is in the paper (2026-11-21, on the user's request "논문에 반영"). The runner, the notebook and the
protocol are on main (PR92, merged 2026-11-22); the paper section and the result tables come with PR91, into
which main was merged once to resolve a conflict in this file. Next, when the user asks: confirm the wording of
the new section, compile both versions on Overleaf, fix the title and authors.

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

## Requirement delta — manual location boxes for the SFT training photos (2026-11-16)

- SFT result (L4, `sft_v1`): LoRA Joint .971 (direct) / .967 (adaptive) vs base .292. Tools add nothing and the crop box collapsed to one corner (341/365). Details in `docs/agent_sft_lora.md`.
- User decision: keep the agent direction and label location by hand. Location GT already exists only on the evaluation side (gold 133, dev missing-cap 24); the 300 anomalous training photos had none.
- Done here: `scripts/sft_train_location_annotation.py`, its screen, `notebooks/lvlm_sft_train_location_annotation_colab.ipynb` (CPU), tests, `docs/sft_train_location_annotation.md`. Target list fixed in `reports/sft_train_location_annotation_preparation_2026-11-16/manifest.json`.
- Next action (user): run the Colab and box the 300 photos; send `sft_train_location_annotation_results.zip`.
- Then: check the 9-cell distribution (gold "always center" = 72/133). If it is not collapsed, rebuild the SFT data with these boxes and retrain. Evaluate the automatic-box and manual-box adapters on gold together, once. Not built yet.
- Not done: retraining, gold evaluation, GRPO.

## Requirement delta — add polymer corrosion to the SFT set (2026-11-16)

- The 300 manual boxes are done and verified (all boxed, 348 boxes). Baselines: center in 159/300, class mode 203/300; on gold "always center" 72/133. Details in `docs/sft_train_location_annotation.md`.
- Found: gold v2 has 24 polymer-shackle defect photos (`val/rust`), but `sft_v1` trained on no polymer defect because supervised train names it `corrosão`.
- User decision: add 60 train `corrosão` photos to the next SFT set with the Q2 answer "Rust / corrosion.". Evaluation labels and gold are unchanged. The user boxes these 60 as well.
- Done here: `select_train_images(..., extra_classes=[POLYMER_CORROSION])` (the sft_v1 600 do not move), the annotation tool accepts `corrosao`, one result ZIP per run folder, `notebooks/lvlm_sft_train_location_annotation_polymer_colab.ipynb`.
- Next action (user): box the 60 polymer photos; send `sft_train_location_annotation_sft_v2_polymer_results.zip`.
- Then (not built): SFT set v2 = 660 photos with manual boxes for the 360 anomalous ones, retrain, and evaluate the automatic-box and manual-box adapters on gold together, once.

## Requirement delta — SFT set v2 with human boxes (2026-11-16)

- Polymer labels: 40 of 60 photos are boxed in the ZIP of 2026-11-16 21:02; 20 remain. The user finishes them in the same Colab (it resumes at photo 41).
- Done here: `build_agent_sft_data.py --human-boxes ... --add-polymer-corrosion` (protocol `agent_sft_train_v2_human_boxes`), `build_agent_sft_notebook.py --variant sft_v2_human`, `notebooks/lvlm_agent_sft_lora_human_boxes_colab.ipynb`, tests. Design and limits in `docs/agent_sft_lora.md`.
- Decisions taken as defaults (not asked): one crop per photo = the largest human box, sides grown to 0.22; Q3 answer = cell of that box centre; normal photos never crop; unlocatable photos leave training; the labels are read from the two Drive label folders.
- Verified locally: real ZIPs + the 300 real boxes give 660 photos / 5,040 samples / 360 crops; unfinished labels are refused. Not verified: any GPU step, a real Colab run of the new notebook.
- Next action (user): finish the 20 polymer photos, then Run all in the new notebook on L4 (about 6-7 h training + 2 h dev240 evaluation, estimated from `sft_v1`).
- Then: check on dev240 that Q1/Q2 hold and that crop boxes are no longer one constant box. Gold evaluation of the two adapters is still one later step and needs the frame-overlap filter and the 9-cell Q3 format first. Not built.
- 2026-11-21: the gold step is built and was run once (PR92, protocol
  `docs/agent_sft_gold_eval.md`). Its results and the paper section are described under
  "Fine-tuning section" above.
