# Tool-choice agent LoRA SFT (follow-up, dev240 only)

Notebook: `notebooks/lvlm_agent_sft_lora_colab.ipynb`. Pick an L4 or A100 runtime (20 GiB or more), then **Run all** once. The notebook builds the data, trains, runs both evaluations and downloads one ZIP. This is a follow-up to the [training-free pilot](agent_tool_pilot.md). It does not change the frozen paper, gold v2 or dev240.

## Why

The training-free pilot (InternVL3.5-8B, L4, 2026-11-09) showed that the model can call the tools but gains nothing from them:

- Q2 macro-F1 was .431 for `direct_reference` and .328 for `adaptive_tools`.
- 44% of the adaptive Q2 answers were invalid, mostly `"A."` with a trailing period.
- Q3 presence was about .5 under every condition.

AgentIAD reports the same pattern: tools help only after SFT/RL. This step teaches the tool policy with LoRA SFT.

## Decisions (user, 2026-11-09)

| Item | Choice |
|---|---|
| Model / GPU | Qwen3.5-4B, BF16 LoRA, fits an L4. No 4-bit or other quantization. |
| Crop boxes | Automatic: DINOv2-base patch tokens of the target vs its different-date normal reference (1 − best cosine), boxed by `vision_expert.select_roi_box`. No human labels. |
| Rationales | Template from GT. Implemented as JSON actions only: the eval parser rejects any text outside the JSON object, so a sentence would make every trained answer invalid. |

## Training data (`scripts/build_agent_sft_data.py`)

- **Source.** InsPLAD supervised `train`, the five dev240 assets. Default is 60 images per class: missing-cap, LRS rust, vari-grip bird-nest and rust, yoke rust, and normals for all five assets.
- **Held out before sampling:**
  - every dev240 image and its fingerprint (size + CRC32)
  - **every crop of the same drone photo** (same flight date + `DJI_nnnn`)
  - unsupervised `test` fingerprints (this includes all gold v2 images)
  - fixed references and duplicates
  - polymer `corrosão` (not merged with rust)
- **Why not split by date.** dev240 covers all 35 flight dates of the train split, so a date-disjoint split is impossible.
- **Overlap that remains.** Many defect photos share a drone frame with unsupervised test images. A model trained here must not be scored on gold v2 without first removing gold images that share a frame with the training set.
- **References.** Different-date DINOv2 Top-1 (`build_retrieved_refs.py`), the same rule dev240 uses.
- **Trajectories.** Built with the eval runner's own `prompt_for` and history strings. The tests replay every trajectory through `run_agent_tool_pilot.episode` and check that it produces identical prompts.

| Case | Steps |
|---|---|
| anomalous | reference → crop(auto box) → finish(GT) |
| normal, patch difference ≥ the asset's normal median | reference → crop → finish(GT) |
| normal, patch difference below that median | reference → finish(GT) |

- **Q3 labels.** For anomalous images, the Q3 answer is the 3×3 cell containing the auto-box centre, so it is noisy.
- **Audit.** `boxes.csv` and `centred_fallback_boxes` in `build_summary.json` record boxes and flat-map fallbacks.
- **Validation split.** 5% of images (by hash) are held out for validation loss.

## Training (`scripts/train_agent_sft_lora.py`)

- **What one sample is.** One agent step. The input comes from the eval adapter's own `apply_chat_template` call (`enable_thinking=False`).
- **Loss.** Only on the action JSON plus `<|im_end|>`.
  - The LM head runs only on those positions (`logits_to_keep`). Full-length logits over the ~248k-token vocabulary needed 2–3 GB per sample and pushed the L4 to its memory limit (2026-11-13). The loss value is the same; a tiny Qwen3.5 check gave identical numbers.
- **LoRA settings.**
  - Applied to every language-model `Linear`; the vision tower is frozen.
  - r 16, α 32, dropout .05.
  - AdamW lr 1e-4, linear warmup 3% then decay, grad accumulation 8, 1 epoch.
  - Gradient checkpointing.
- **Resume.** A checkpoint is written every 50 optimizer steps, so a disconnected run continues where it stopped.

## Evaluation

- `run_agent_tool_pilot.py` runs twice on the same dev240, with the same 4 conditions, max 3 calls and scoring:
  - Qwen3.5-4B base
  - the same model with `--adapter-dir`
- **What the adapter run records.**
  - The adapter bytes are bound into the run signature.
  - `run_metadata.training` is `true` and the adapter file hashes are listed.
- **Main comparison:** `adaptive_tools` (LoRA) vs `direct_reference` and `adaptive_tools` (base). `fixed_evidence` and `matched_reask` stay as controls.
- **Caveats:**
  - dev240 has been explored repeatedly. It has no spatial GT, so Q3 is scored on presence only.
  - A gain here is a dev result, not a generalisation claim.
- **Not done:** GRPO, gold evaluation, and a new holdout.

## Result (2026-11-16, L4, dev240, run `sft_v1`)

Training finished 587/587 steps; validation loss went from .826 to .134. Both evaluations completed with 0 row errors.

| Condition | Q1 F1 | Q2 macro-F1 | Joint | Q3 presence BAcc | Invalid episodes | Calls |
|---|---|---|---|---|---|---|
| base, direct_reference | .375 | .421 | .292 | 0 (all invalid) | 419 | 720 |
| base, adaptive_tools | .463 | .494 | .292 | .144 | 313 | 720 |
| LoRA, direct_reference | .987 | .963 | .971 | .988 | 0 | 720 |
| LoRA, fixed_evidence | .987 | .969 | .975 | .988 | 0 | 720 |
| LoRA, adaptive_tools | .979 | .960 | .967 | .976 | 1 | 1,805 |

- **Fine-tuning helps a lot.** The base scores are low mostly because of format: 419 of 720 direct answers were prose, not JSON.
- **Tools add nothing.** LoRA adaptive vs LoRA direct: every bootstrap interval includes 0, and adaptive is slightly lower with 2.5 times the calls.
- **The learned tool policy is degenerate.**
  - Cropping equals the anomaly decision: all 115 anomalous photos get reference then crop; 119 of 125 normal photos get reference only.
  - 341 of 365 crop boxes are the same corner `[0, 0, .221, .221]`. That box was 15 of the 300 training boxes. The automatic boxes cannot be predicted from the photo, so the policy fell back to the most frequent one.
  - Q3 cell on anomalous photos: adaptive says upper-left 105/115, direct says center 92/115. dev240 has no location GT, so this is not scored.
- **Inflation check.** No dev photo shares a drone frame with training. 60 dev photos have a same-date, same-asset, same-label training photo within 5 frames; accuracy is the same for them (58/60) as for the other 180 (175/180). dev240 still comes from the same supervised-train pool and dates as the training photos.
- **Supported claim:** fine-tuning on 600 labelled photos nearly solves Q1/Q2 on dev240. **Not supported:** that tool use helps, or that location is learned.
- **Known logging bug:** `sec_per_sample` in `train_log.jsonl` is cumulative time over the last 80 samples. Real speed was about 4.5 s per sample. Results are unaffected.
- **Next:** manual boxes for the 300 anomalous training photos, [sft_train_location_annotation.md](sft_train_location_annotation.md).

## Second run: human boxes (`sft_v2_human`, prepared 2026-11-16, not run yet)

`sft_v1` learned no location: its crop box was a by-product of an automatic difference map, and 341 of 365 dev240
crops were the same corner. The second run changes the training data only. Model, LoRA settings, prompts, parser and
the dev240 evaluation are the same.

| | `sft_v1` | `sft_v2_human` |
|---|---|---|
| Photos | 600 | the same 600 + 60 polymer `corrosão` |
| Crop box, anomalous | automatic DINOv2 patch difference | the largest human box, each side grown to at least 0.22 |
| Q3 answer, anomalous | cell of the automatic box centre | cell of that human box's centre |
| Normal photos | crop on about half (score above the asset median) | never crop: reference, then finish |
| Protocol | `agent_sft_train_v1` | `agent_sft_train_v2_human_boxes` |

- Build: `build_agent_sft_data.py --add-polymer-corrosion --human-boxes <label run folder> ...`. A label run is a
  Drive folder of the box-label Colab (`manifest.json` + `annotations.json`). The builder refuses a run with an
  undecided photo, a label whose photo SHA differs, and label sets that do not equal the selected anomalous photos.
  A photo marked unlocatable or disputed leaves training. The label file SHAs are part of the build settings.
- Polymer `corrosão` photos are asked and answered as "Rust / corrosion." in training only (user decision
  2026-11-16). Evaluation labels and the slug contract do not change.
- Checked locally on the real ZIPs with the 300 real boxes (no model call; the reference search used fake
  embeddings, and 20 unlabelled polymer photos had placeholder boxes in a scratch copy): 660 photos, 5,040 step
  samples, 360 crops, every crop contains its human box, 97 small boxes grown, all targets pass the eval parser.
  With the labels as they are on Drive now (polymer 40/60) the builder stops with "20 of 60 photos are not labelled".
- Notebook: `notebooks/lvlm_agent_sft_lora_human_boxes_colab.ipynb`. It checks both label folders before any GPU
  work, trains, evaluates the new adapter on dev240, and shows the `sft_v1` base and LoRA rows from Drive without
  rerunning them. It also counts distinct crop boxes, the first sign of whether location was learned.
- Limits. Three things change at once (box source, polymer class, normal-photo policy), so a difference from
  `sft_v1` cannot be assigned to one of them. dev240 has location GT for 24 photos only, so location accuracy is not
  measured here. A crop is requested exactly when the photo is anomalous in the training data, so this run cannot
  show that the crop tool adds accuracy; what it can show is whether the agent points at the defect.
