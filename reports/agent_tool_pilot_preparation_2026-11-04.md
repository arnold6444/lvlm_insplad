# Tool-choice pilot preparation — 2026-11-04

## Confirmed delivery

One Colab notebook and a dev-only runner are prepared for a training-free adaptive reference/crop loop. This report contains **implementation verification, not neural model performance**. No Colab inference, SFT or GRPO has been executed for this new pilot.

Source execution pin: `150aa27f8840870729b9f842ac62c5b79d1e35c6`. The notebook checks exact HEAD, remote and tracked-tree cleanliness, then runs the committed source. The notebook is delivered in a later commit so its source pin is immutable.

## CPU verification

- `python -m pytest tests/test_agent_tool_pilot.py tests/test_gold_v2_q2_q3_agent_dev.py tests/test_dev_prompt_diagnostic.py -q --basetemp=.pytest-tmp`: **36 passed**.
- After final provenance/reference compatibility fixes, `python -m pytest tests/test_agent_tool_pilot.py -q --basetemp=.pytest-tmp`: **20 passed**.
- Notebook: four code cells compile, the full source pin exists and includes the runner, and all output cells are empty.
- `git diff --check`: clean.
- Tests execute real image cropping and file/checkpoint handling with injected model responses, not a real VLM. Test metadata says `injected_test_backend`.

Covered behaviors: strict JSON/coordinates, actual requested crop, early finish, one reference/crop per question, maximum calls, matching re-ask budget, question independence and GT isolation, dev-only sampling, no spatial-GT claims, no retry of invalid responses, retry of inference errors, model-load abort, unsafe extraction refusal, changed-input/image refusal, and zero neural calls on a fully reused run.

## GPU run still required

Use one >=20 GiB L4/A100-class GPU. Start with 40 images (at most 960 LVLM calls); after inspecting the functionality gate and class-level results, the same settings may be run on 240 (at most 5,760 calls). Actual speed, format reliability, accuracy gains and costs remain unmeasured. There is no expected-score range backed by data for this method.

These two runs finish a bounded exploratory inference comparison. They do not complete training, an untouched holdout evaluation or Q3 spatial validation. Dev lacks defective spatial GT, so Q3 location and Q1/Q2/Q3-location joint are null. The historical 11.7%/37.9% location joint must not be compared with this pilot's presence joint.

## Scope and preservation

Original gold/dev files and paper artifacts are not modified. The separate worktree/branch is `lvlm_insplad_agent_pilot` / `codex/agent-tool-pilot`. Existing uncommitted paper and offline joint-scoring changes remain in `lvlm_insplad_fresh`.

## Publication status

GitHub upload was requested by the user and attempted after verification. Automatic approval review rejected `git push -u origin codex/agent-tool-pilot`: approval is required, but the current approval policy is Never. The branch does not exist remotely (`git ls-remote --heads origin codex/agent-tool-pilot` returned no match), so no PR or usable remote Colab link is claimed. Local commits and notebook are ready for manual publication. A generated test scratch directory remains untracked because automatic review also denied its deletion; it is not part of any commit.

```powershell
git -C C:\Users\user\Desktop\lvlm_insplad_agent_pilot push -u origin codex/agent-tool-pilot
```
