# InsPLAD-QA: an LVLM benchmark for power-line inspection

InsPLAD-QA is an MMAD-style question-answering benchmark built on the public InsPLAD UAV dataset. It tests whether vision-language models detect a defect (Q1), name it (Q2) and say whether a defect region is present (Q3), and whether those answers agree. Every model uses the same QA, prompt, output schema and evaluator.

**Status (2026-10-24): final.** The scope is frozen, all planned experiments are done, and the results are written up in the paper draft.

- **Paper (English/Korean, LaTeX and Word):** [`reports/paper_draft_2026-10-23/`](reports/paper_draft_2026-10-23/README.md)
- **All results in one place:** [`reports/paper_tables/latest_results.md`](reports/paper_tables/latest_results.md)
- **Independent check of every number in the paper:** [`VERIFICATION_2026-10-24.md`](reports/paper_draft_2026-10-23/VERIFICATION_2026-10-24.md)

## Main findings (gold v2: 246 images, 534 questions)

- **Image-only baselines.** Q1 F1 ranges from .555 to .718 and Q2 macro-F1 from .360 to .505 across four open LVLMs and one API pipeline.
- **Normal reference.** It raises Q1 F1 for every open model, for example InternVL3.5-8B from .597 to .794. For MiniCPM-V-4.5, however, balanced accuracy falls (.656 → .587) because it flags most normal images. Its effect on consistency is mixed.
- **Anomaly-detector cues.** They mostly raise defect-type F1 but lower anomaly-detection F1. No cue improves both for every model.
- **Verification agent (InternVL3.5-8B).** The agent was chosen on a disjoint dev set and run once on gold.
  - Q2 macro-F1 rises from .522 to .640 (+.118, paired 95% CI [+.043, +.199]).
  - Region-presence balanced accuracy rises from .551 to .828.
  - False alarms on normal images fall from 93/107 to 16/107.
- **Matched-call control.** An equal-call re-ask without the agent's definitions and staging trails the agent by .164 Q2 macro-F1 and .203 presence BAcc. Both CIs exclude zero.
- **Cost.** Missed defects rise from 4/139 to 27/139, mostly missing insulator caps (the agent names that defect in 5 of 27 images). Dev-only follow-ups with higher resolution, other LVLMs, anomaly detectors and part-evidence tools did not recover them. Only an asset-specific supervised classifier did.

## Benchmark design

| Part | Source | Images | Use |
|---|---|---|---|
| Track A, general anomaly detection | `unsupervised_anomaly_detection.zip`, official `test` | 6,481 (17 assets) | zero-shot and one-shot (normal reference) |
| Track B, structured fault understanding | `supervised_fault_classification.zip`, official `val` (no test split) | 6,417 (5 assets) | anomaly label + defect class, Joint Accuracy |
| Aligned Core | unsup test ∩ sup val ∩ identical SHA-256 ∩ same binary GT | 1,378 | same image bytes across tracks |
| Gold v2 (frozen) | Aligned Core, manually reviewed | 246 images / 534 questions | all paper results |
| Dev | supervised `train`, seed 42, leakage-checked | 240 images / 480 Q1–Q2 questions | every method choice |

- Three Track A assets are normal-only in test: `damper-preformed`, `spacer`, `yoke`. Six exact-duplicate images have conflicting binary GT and are left out of cross-track comparisons.
- One-shot references come only from the same asset's unsupervised train normals (seed 42). Candidates whose size+CRC32 fingerprint appears in test are excluded. The retrieved reference is the DINOv2-base cosine nearest neighbour.

### Label policy
- `good`/`normal` → `normal`; every defect folder → `anomalous`.
- `corrosão` → `corrosao` (slug only, never merged with `rust`).
- The integers in `image_list_val.txt` are asset-local class indices, not binary labels, so semantic labels come from folder names.

### Metrics
- Anomaly: Balanced Accuracy, F1, Precision/Recall, TP/FP/TN/FN, Invalid Output Rate, per asset and asset-macro.
- Structured: defect BAcc, Macro-F1, Joint Accuracy, False Defect Rate, Consistency Error, Invalid Output Rate.
- An invalid answer counts as FN for an anomalous image and is reported apart from FP for a normal image (`metrics.anomaly_classification_summary`).

## Models

One model per model line, unquantized on an L4/A100-class GPU (at least 20 GiB VRAM; the HF runners refuse smaller GPUs):

- `Qwen/Qwen3.5-4B` (thinking disabled)
- `OpenGVLab/InternVL3_5-8B` (official chat path, 6 visual tiles)
- `openbmb/MiniCPM-V-4_5` (official `chat()`, greedy)
- `llava-hf/llava-onevision-qwen2-7b-ov-hf`
- Luxia two-stage API pipeline (document vision → chat), image-only only

## Repository map

| Path | Contents |
|---|---|
| `src/insplad_benchmark/` | dataset parsing, QA utilities, metrics, model adapters, retrieval |
| `scripts/` | benchmark build, runners, evaluation and notebook builders |
| `notebooks/` | Colab notebooks, listed below by paper section |
| `results/latest/` | published leaderboards and metrics, each with a `github_publish_manifest.json` (SHA-256 of the committed files) |
| `reports/paper_draft_2026-10-23/` | the paper |
| `reports/paper_tables/` | result tables, including `latest_results.md` |
| `reports/*_evaluation_*/`, `reports/*_research_*/`, ... | one summary (`analysis.md`) per experiment run |
| `reports/*.md`, `reports/*.csv` | dataset structure, overlap and QA validation records |
| `docs/` | experiment protocols, indexed in [`docs/README.md`](docs/README.md) |
| `configs/`, `samples/` | label mappings, locked configurations, 20-row sample QA |

Raw per-call logs and the Colab result ZIPs are not in git. They are archived with SHA-256 hashes, which are recorded in the manifests and the analysis files, and are available on request.

## Notebooks by paper section

Open a notebook in Colab on an L4/A100 runtime, put `unsupervised_anomaly_detection.zip` and `supervised_fault_classification.zip` anywhere in Google Drive `MyDrive`, and choose **Run all**. Each run writes to its own Drive folder and resumes from saved calls.

| Paper section | Notebooks | Results |
|---|---|---|
| Benchmark build (§3) | `build_explainable_qa_gold_v2_colab`, `lvlm_gold_v2_dev_set_colab` | `gold_v2_final`, `gold_v2_dev` |
| Image-only baselines (§4) | `lvlm_gold_v2_baselines_colab`; four-option Q3: `lvlm_gold_v2_loc_box_annotation_colab`, `lvlm_gold_v2_loc_mmad_eval_colab`; MiniCPM greedy re-run: `lvlm_gold_v2_minicpm_greedy_colab` | `gold_v2_baseline`, `gold_v2_loc_mmad`, `gold_v2_minicpm_greedy` |
| References and expert cues (§4) | `lvlm_gold_v2_reference_ablation_colab` (+ `_5lvlm_finalize`, `_hf4_resume`, `_luxia_resume`), `lvlm_gold_v2_reference_expert_colab`, `lvlm_vision_expert_smoke(_resumable)_colab` → `lvlm_gold_v2_vision_expert_selection_colab` → `lvlm_gold_v2_vision_expert_full_final/resume_colab` | `gold_v2_reference`, `gold_v2_reference_expert`, `gold_v2_vision_expert_*`, `gold_v2_consistency_metrics` |
| Agent design on dev (§5) | `lvlm_gold_v2_dev_baselines_colab`, `lvlm_gold_v2_consistency_agent_colab` (re-ask pilot), `lvlm_gold_v2_prompt_diagnostic_colab`, `lvlm_gold_v2_offline_agent_colab`, `lvlm_gold_v2_verify_agent_dev_colab`, `lvlm_gold_v2_q2_q3_agent_dev_colab`, `lvlm_gold_v2_q2_q3_refined_dev_colab` | `gold_v2_dev_baselines`, `gold_v2_offline_agent`, `gold_v2_verify_agent_dev` |
| Agent on gold (§5) | `lvlm_gold_v2_main_rerun_colab` (retrieved-reference direct answers), `lvlm_gold_v2_verify_agent_gold_colab`, `lvlm_gold_v2_q23_candidate_colab` (+ `_reaggregate`), `lvlm_gold_v2_matched_call_colab`; Q3 position re-scoring (CPU, no inference): `scripts/score_gold_v2_q3_position.py` | `gold_v2_matched_call` (its first three policies equal gold run `0c577ffe3bdd`), `gold_v2_q3_position` |
| Missing-cap follow-ups, dev only (§6, appendix) | `lvlm_gold_v2_q23_zoom_prompt_dev_colab`, `lvlm_gold_v2_q3_multiregion_colab`, `lvlm_dev_missing_location_annotation_colab`, `lvlm_q2_conflict_evidence_dev_colab`, `lvlm_q3_gt_crop_diagnostic_dev_colab`, `lvlm_q3_missing_occlusion_dev_colab`, `lvlm_selective_missing_expert_agent_dev_colab`, `lvlm_missing_presence_independent_dev_colab`, `lvlm_missing_expert_direct_dev_colab`, `lvlm_missing_ig_location_dev_colab`, `lvlm_missing_ig_numerics_colab`, `missing_component_probe_colab` | summaries under `reports/` |
| Full-track runners (not run for the paper) | `lvlm_baselines_full_colab`, `luxia_baselines_full_colab`, `lvlm_one_shot_plus_colab` | — |

The other dev follow-ups in the paper's appendix (other LVLMs, anomaly detectors, cross-model review, part-evidence and alignment probes) ran from frozen source bundles. Their code and raw logs are archived and not part of this repository. Their summaries are in `reports/pretrained_inspection_research_2026-10-17/`, `reports/q2_cross_model_review_evaluation_2026-10-16/`, `reports/counting_structure_research_2026-10-17/` and the other dated folders linked from `latest_results.md`.

## Setup and tests

```powershell
python -m pip install -e ".[dev]"            # retrieval and HF inference: .[retrieval,hf]
python -m pytest -q --basetemp=.pytest-tmp    # fixtures only; no source ZIPs or GPU needed
```

## Evaluation CLI

```powershell
python scripts/evaluate.py --task anomaly --qa <qa.jsonl> --predictions <pred.jsonl> --regime zero_shot|one_shot|one_shot_plus
python scripts/evaluate.py --task explainability --qa <defect_qa.jsonl> --predictions <pred.jsonl>
```

## Data source

InsPLAD is published on Mendeley Data; [`insplad_google_drive`](insplad_google_drive) holds the official download link. The source ZIPs and images are not committed.
