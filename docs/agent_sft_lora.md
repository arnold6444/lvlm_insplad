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
