# AgentIAD-lite expanded development recheck

This stage rechecks the 50-row AgentIAD-lite result on a larger development cohort before touching the final holdout.

- deterministic dev size: 150
- seed: 42
- the original 50-row smoke is a subset of the 150-row cohort
- binary balance: 75 normal / 75 anomalous
- asset coverage is round-robin balanced through `balanced_binary_sample`
- compared methods:
  - fixed `InternVL3.5-8B + DINOv2 retrieved normal reference`
  - `Agent-RefFirst`: fixed reference decision followed by conditional PatchCore zoom
- `Agent-QueryFirst` is intentionally omitted because the first smoke showed a structural failure and extra compute would not answer the current question
- PatchCore is recomputed on the 150 queries with the same leakage-safe per-asset train-normal protocol
- the dev percentile used for routing is method-selection only; final holdout requires train-normal score calibration

After this run, the remaining 1,228 Aligned Core rows stay untouched as the final holdout if the 150-row protocol is frozen.
