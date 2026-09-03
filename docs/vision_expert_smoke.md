# Vision Expert Screening Smoke

## Scope

This experiment starts after the completed Aligned Core baseline. It fixes one deterministic 50-row cohort and screens only:

- DINOv2 retrieved normal reference
- PatchCore ROI
- Dinomaly ROI
- AnomalyCLIP ROI
- best smoke ROI + DINOv2 retrieved normal reference

Qwen3.5-4B and InternVL3.5-8B are used for screening. RAG, AgentIAD-lite, full 1,378-row inference, MiniCPM, LLaVA, and Luxia are intentionally outside this first stage.

## Run in Colab

Open `notebooks/lvlm_vision_expert_smoke_colab.ipynb` from branch `feat/vision-expert-smoke-colab` and run from top to bottom on an L4/A100-class runtime.

Colab Secrets:

- `GITHUB_TOKEN`: read-only access to this private repository
- `HF_TOKEN`: optional, recommended for Hugging Face downloads

Google Drive must contain exactly one copy of:

- `unsupervised_anomaly_detection.zip`
- `supervised_fault_classification.zip`

If duplicate filenames exist, set `DRIVE_DATA_DIR` in the configuration cell.

Persistent outputs are written under:

```text
MyDrive/lvlm_insplad_results/vision_expert_smoke_v1/
```

The notebook checkpoints predictions and reuses complete ROI artifacts. A stopped Colab session can therefore resume by rerunning cells from the top.

## Selection and paper protocol

The smoke winner is chosen by the mean anomaly F1 across Qwen and InternVL; balanced accuracy and invalid-output rate break ties. This 50-row set is a method-selection set, not the final paper score.

After selection:

1. freeze the winning vision expert and evidence format;
2. evaluate first on the 1,328 Aligned Core rows not used by the smoke cohort;
3. if the gain reproduces, add defect-knowledge RAG;
4. compare an AgentIAD-lite router against the fixed best evidence combination;
5. run the final winner on all 1,378 rows and the remaining LVLMs for benchmark comparability.
