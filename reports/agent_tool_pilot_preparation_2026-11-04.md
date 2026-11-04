# Tool-choice pilot preparation — 2026-11-04

## Confirmed delivery

One self-contained Colab notebook and a dev-only runner are prepared for a training-free adaptive reference/crop loop. One Run all executes dev240 and all four controls, displays scores and downloads results. This report contains **implementation verification, not neural model performance**. No Colab inference, SFT or GRPO has been executed for this method.

Current source execution pin: `c9b3f10a996f52f3bec20904da006533de05eb23`. The notebook embeds 105 tracked source/config files from that commit, validates the archive and per-file SHA-256 hashes, and refuses changed existing source. It needs no git repository, remote source access, token or GitHub publication. The delivered notebook is 589,701 bytes with empty output cells.

## CPU verification

- `python -m pytest tests/test_agent_tool_pilot.py tests/test_gold_v2_q2_q3_agent_dev.py tests/test_dev_prompt_diagnostic.py -q --basetemp=.pytest-tmp`: **36 passed**.
- After final provenance/reference compatibility fixes, `python -m pytest tests/test_agent_tool_pilot.py -q --basetemp=.pytest-tmp`: **20 passed**.
- After the one-run/self-contained update, the same targeted file: **22 passed**, including no-Git standalone source import, source tamper refusal and diagnostic ZIP delivery on failure.
- Notebook: four code cells compile. Its embedded source was independently unpacked without `.git`; the extracted runner reports the exact source commit through its validated receipt. All output cells are empty.
- `git diff --check`: clean.
- Tests execute real image cropping and file/checkpoint handling with injected model responses, not a real VLM. Test metadata says `injected_test_backend`.

Covered behaviors: strict JSON/coordinates, actual requested crop, early finish, one reference/crop per question, maximum calls, matching re-ask budget, question independence and GT isolation, dev-only sampling, no spatial-GT claims, no retry of invalid responses, retry of inference errors, model-load abort, unsafe extraction refusal, changed-input/image refusal, and zero neural calls on a fully reused run.

## GPU run still required

Use one >=20 GiB L4/A100-class GPU. Run all once on 240 images (at most 5,760 LVLM calls); no separate 40-image run or setting change is required. Actual speed, format reliability, accuracy gains and costs remain unmeasured. There is no expected-score range backed by data for this method.

This one run finishes a bounded exploratory inference comparison. It does not complete training, an untouched holdout evaluation or Q3 spatial validation. Dev lacks defective spatial GT, so Q3 location and Q1/Q2/Q3-location joint are null. The historical 11.7%/37.9% location joint must not be compared with this method's presence joint.

## Scope and preservation

Original gold/dev files and paper artifacts are not modified. The separate worktree/branch is `lvlm_insplad_agent_pilot` / `codex/agent-tool-pilot`. Existing uncommitted paper and offline joint-scoring changes remain in `lvlm_insplad_fresh`.

## Publication status

GitHub upload remains unperformed after the earlier automatic approval rejection; no retry or alternate remote-write mechanism was used. This no longer blocks notebook execution: upload the local notebook file in Colab and use Run all. Model/dependency downloads, GPU availability and authorization to the existing Drive are still required. A generated test scratch directory remains untracked and is not part of any commit.
