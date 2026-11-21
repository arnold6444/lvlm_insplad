# Q1 + Q2 + Q3 location joint accuracy (2026-11-03)

Post-hoc analysis of existing InternVL3.5-8B answers. No new model inference or training.

## Definition and scope

A success requires correct Q1 (normal/anomalous), correct Q2 (normal/rust/bird-nest/missing-cap), and correct Q3 on the same image. On defective images, the predicted single 3x3 cell must contain the center of at least one annotated defect box. On normal images, Q3 must report no region. This is the existing position-hit rule, not exact multiple-region agreement or bounding-box overlap.

All policies use the same 240 images: 107 normals plus 133 defective images with spatial GT. Six defective images without spatial GT are excluded from every policy, not treated as wrong or right. This is a gold subset, not dev240 and not the original 42-question Q3 subset.

The baseline already uses a retrieved different-date normal reference. These results do not measure improvement over an image-only baseline and cannot rank all models/cue configurations from earlier, different Q3 protocols.

## Results

| Policy | All correct / 240 | Joint accuracy | Normal / 107 | Defective / 133 |
|---|---:|---:|---:|---:|
| Direct questions (baseline) | 28 | 11.67% | 10 | 18 |
| Equal-call re-ask | 43 | 17.92% | 30 | 13 |
| Existing verification pipeline | 91 | 37.92% | 71 | 20 |
| Revised candidate | 94 | 39.17% | 75 | 19 |

Existing pipeline versus baseline: +63 successes, +26.25 percentage points; paired image bootstrap 95% CI [+20.42, +32.50] points.

Candidate versus baseline: +66 successes, +27.50 percentage points (3.36 times the baseline rate); paired bootstrap 95% CI [+21.25, +34.17] points.

Candidate versus existing pipeline: +3 successes, +1.25 points; CI [-1.67, +4.58] points. The candidate is numerically highest among these four policies, but superiority over the existing pipeline is not established. CIs are exploratory, unadjusted for multiple comparisons, and do not remove historical gold reuse or post-hoc metric selection.

## Interpretation

Of the candidate's 66 extra successes, 65 are normal images; only one is a defective image. Defective-image joint accuracy is 18/133 (13.53%) for baseline, 20/133 (15.04%) for the existing pipeline, and 19/133 (14.29%) for the candidate. Missing-cap joint success is 0/27 in all four policies.

Thus the large aggregate gain mostly reflects correct normal decisions, not broad improvement in identifying and locating defects. The metric measures structured-answer correctness, not the faithfulness of internal reasoning. The cohort has 107 normal / 133 defective images; results do not estimate field prevalence.

## Verification and reproduction

- Scorer reproduces all four published Q1/Q2 joint rates, Q1/Q2/Q3-presence joint rates, and Q3 position-hit counts before writing outputs.
- An independent PowerShell calculation directly from source answers, labels and position GT reproduced 28, 43, 91 and 94 successes.
- `run_metadata.json` records input and scorer SHA-256 hashes, excluded IDs and scope.
- Reproduce from repository root: `python scripts/score_gold_v2_joint_location.py` (standard library only).
- New outputs are in this directory; frozen QA, original predictions and paper drafts are unchanged by this analysis.
