# MiniCPM-V-4.5 greedy re-run on gold (reference conditions)

## Why
MiniCPM-V-4.5's `chat()` samples by default (temperature 0.7). The fix in PR #47 came after its gold baseline and reference runs, so the MiniCPM rows of the paper's Tables 2–3 are single sampled runs. This notebook re-runs exactly those inputs with greedy decoding.

## Run
Open `notebooks/lvlm_gold_v2_minicpm_greedy_colab.ipynb` in a fresh **L4/A100** Colab runtime → **Runtime → Run all** → authorize Drive → return `gold_minicpm_greedy_results.zip`.
- No token or manual steps are needed. The notebook clones public `main` and runs only if every imported code file, plus the reference manifest, matches the tested git blob IDs.
- Nothing is published: the notebook removes `GITHUB_TOKEN` from the environment and never calls the publish script.

Output: `MyDrive/lvlm_insplad_results/gold_v2_minicpm_greedy_v1/`. Re-running the inference cell resumes the same run, and only failed calls are retried.

## Protocol
| Condition | Images | Prompt source |
|---|---|---|
| `image_only` | target | `run_gold_baselines.prompt_for` |
| `normal_reference` | fixed normal, target | `run_gold_v2_reference_ablation.prompt_for_condition` |
| `positive_reference` | retrieved defect example, target | same |
| `normal_positive_reference` | fixed normal, defect example, target | same |

- The defect examples are the published `results/latest/gold_v2_reference/reference_manifest.csv`. It is re-checked against the frozen QA (same target and fixed normal) and requires label-free selection.
- The prompt functions and image order have not changed since the original runs (September 15–16). The MiniCPM adapter changed only its decoding: `sampling=False`, `num_beams=1`, `repetition_penalty=1.0`, `max_new_tokens=80`.
- All 534 gold questions are asked per condition (Q1/Q2 246 each, Q3 42), for 2,136 calls (about 1–1.5 h on an L4).
- Invalid outputs count as wrong, as in the original evaluation. Metrics use `run_gold_v2_consistency_metrics`. Fed the published image-only predictions, these functions reproduce the published MiniCPM image-only row exactly.

## Outputs
- `leaderboard.csv`: per condition Q1 F1, Q2 macro-F1, joint accuracy, consistency error, false defect rate, Q3 accuracy and invalid rates, plus Q1 accuracy, balanced accuracy, precision, recall, specificity, TP/FP/TN/FN and invalid normal/anomalous counts (kept apart from FP/FN), and Q2 accuracy and defect balanced accuracy.
- `metrics_by_asset.csv` (same columns per asset, with class support) and `metrics_asset_macro.csv` (asset-macro averages).
- `comparison_vs_sampled.csv`: greedy values against the published sampled values.
- `significance.csv`: image-only versus each reference condition, paired bootstrap 2,000.
- Raw `steps.jsonl`, `image_manifest.csv`, `run_metadata.json` and `run_status.json`.

A run is complete only when all four conditions have zero errors. The run stops on changed QA, manifest, code, image bytes or a reused-step identity mismatch; use a new folder in that case.

The notebook passes a fresh `--invocation-id` to every run and keeps its exit code. The check cell accepts results only when that exit code is 0 and `run_status.json` and `run_metadata.json` carry the same invocation ID, so a re-run that fails early cannot pass on an earlier run's files. The runner loads the model only on a GPU with at least 20 GiB VRAM; a fully cached resume never loads it.
