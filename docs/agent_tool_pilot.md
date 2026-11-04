# Training-free tool-choice pilot

This is a bounded follow-up authorized on 2026-11-04, not an AgentIAD reproduction or a completed model experiment. It adds no training, web search, detector, gold execution or automatic GitHub result publication.

## Run and stopping rule

Use `notebooks/lvlm_agent_tool_pilot_colab.ipynb` on one L4/A100-class GPU with at least 20 GiB VRAM. Default model: existing native, unquantized InternVL3.5-8B adapter, greedy decoding, 80 output tokens, total visual tile budget 6. The user's local 6 GiB GPU is insufficient under the project contract. Colab GPU availability is not guaranteed.

1. Run the notebook with `N_IMAGES=40`. Inspect format validity, errors, chosen crops/references, misses and call costs. Zero inference errors and adaptive format validity >=90% are only a functionality gate, not evidence of performance improvement.
2. If the pilot is useful, keep settings unchanged and run the same notebook with `N_IMAGES=240` in its separate automatically named output folder. These 240 include the same 40; the two scores are not independent replications. Report the result and stop this pilot rather than repeatedly optimizing prompts.
3. Q3 spatial GT is missing on this dev set. A future location evaluation requires verified position annotations and a locked common scoring rule. Final scientific generalization also needs a new independent evaluation set. Two runs finish only the inference pilot, not SFT/GRPO or full explainability validation.

Maximum LVLM calls: 24 per image across four controls (40 images: 960; 240 images: 5,760). This is an upper bound, not a measured runtime. Early adaptive termination lowers calls. Reference preparation, model loading and download costs are separate. There is no reliable expected accuracy or gain before GPU execution; no synthetic performance forecast is used.

## Fixed protocol

Each question runs independently, with a new history. Q1 is freshly evaluated in this new pilot rather than rewritten in older runs. Prompts contain question/options and visible image roles only; GT labels, paths, previous questions' answers and other policies' predictions cannot route actions.

- `direct_reference`: target plus different-date retrieved normal; one JSON final-answer call.
- `fixed_evidence`: target plus the same normal and a fixed center-60% crop; one JSON final-answer call. This is a new fixed-evidence control, not the previously published verification pipeline.
- `adaptive_tools`: initially target only. Strict JSON `reference`, normalized original-frame `crop`, or `finish` action. Python retrieves the prepared reference or crops the actual original pixels, then supplies the updated images and own-question history. Each tool at most once; maximum three calls and the final turn requires finish.
- `matched_reask`: target plus normal, repeated final-answer prompts with the adaptive episode's actual number of calls, including its own previous answer. No new evidence. This controls call count, not every prompt/image-budget difference.

Tools are local Python operations. Reference retrieval uses the existing DINOv2 different-date mapping; no external service or API key is needed. Adaptive cropping is model-specified, not selected from GT. Cropping a small patch can expose details but cannot create detail absent from the original. Optional images share InternVL's total tile budget and can reduce original-target resolution. Prompt, images and available actions differ across controls, so improvements cannot automatically be attributed solely to action selection.

## Inputs and preservation

Reuse the FROZEN `MyDrive/lvlm_insplad_qa/gold_v2_dev` set and original supervised/unsupervised ZIPs. The notebook reuses `gold_v2_dev/refs/retrieved_refs.csv`; if absent, it builds references in a new pilot-only folder, never modifying original QA/results. The runner rejects gold protocols and incomplete/duplicate Q1/Q2 pairs. Output paths must start with `agent_tool_pilot_dev`.

The notebook executes an immutable commit SHA and verifies HEAD and tracked-tree cleanliness before inference. Resume additionally binds code, model specification, questions/options, references, settings, actual target/reference image bytes and every saved prompt/image history. Changed inputs require a new output folder. Successful and invalid responses are reused; errors alone are retried. Invalid JSON, choices, repeated tools or out-of-range boxes count as invalid failures, with no answer fallback or retry-until-correct. An inference error excludes its policy from score comparisons. A model-load failure aborts immediately.

## Evaluation and output

- Q1 F1/BAcc/precision/recall/TP/FP/TN/FN/invalid rate.
- Q2 macro-F1, Q1/Q2 joint, normal false-defect rate and valid-pair consistency error; defect-specific Q2 correctness and asset metrics.
- Q3 presence BAcc, normal false-region and defective false-none rates, plus separately named Q1/Q2/Q3-**presence** joint.
- **Q3 location accuracy and Q1/Q2/Q3-location joint are null.** A wrong cell on a defective image can still count as correct presence. Never compare this presence joint to the historical 11.7%/37.9% location joint.
- Paired exploratory bootstrap CIs (2,000, seed 42), unadjusted for multiple comparisons. Dev is already repeatedly explored.
- Per-call raw output, prompt, images, latency, final action status, per-episode history, QA snapshot, image hashes, source/input identity, GPU/backend metadata and run status.

The downloaded ZIP includes metadata, scores and trace files, not original images or weights. `complete=true` means inference coverage without inference errors, not a successful improvement or a complete spatial evaluation. Test backend outputs are explicitly marked `injected_test_backend` and are never real model results.

CPU verification does not establish model effectiveness. After a Colab run, inspect `run_status.json`, raw actions, class-specific failures and confidence intervals before proposing any SFT/GRPO work or paper claim.

## CLI

```powershell
python scripts/run_agent_tool_pilot.py --dev-qa <dev.jsonl> --refs-csv <retrieved_refs.csv> --sup-zip <supervised.zip> --unsup-zip <unsupervised.zip> --out-dir <agent_tool_pilot_dev40_v1> --n-images 40
```
