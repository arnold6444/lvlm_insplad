# Vari-grip explainable QA smoke

## Scope

- Main category: `vari-grip`
- Source subset: verified Aligned Core sample rows
- Images: 4
  - normal: 1
  - bird-nest: 2
  - rust: 1
- QA per image: 5
- Total QA: 20 multiple-choice questions

Question types:

1. `anomaly_detection`
2. `defect_classification`
3. `defect_localization`
4. `defect_description`
5. `defect_analysis`

Dataset file: `samples/varigrip_explainable_qa_smoke_20.jsonl`

## Ground-truth policy

- `anomaly_detection` and `defect_classification`
  - answer is derived directly from the verified InsPLAD source label
  - `gt_basis = source_label`
  - `review_status = label_verified`
- `defect_localization`, `defect_description`, and `defect_analysis`
  - answer is a label-conditioned draft for smoke-format review
  - these fields are not supplied by the original InsPLAD annotations
  - `gt_basis = label_conditioned_draft`
  - `review_status = pending_visual_review`

Do not promote the latter three question types to benchmark gold until the image itself has been manually checked.

## Manual review checklist

For every `pending_visual_review` item, inspect the actual query image and verify:

- the keyed answer is visibly supported by the image
- the question can be answered from the image without hidden metadata
- only one option is clearly correct
- localization is sufficiently specific but not more precise than the image supports
- description states visible evidence rather than merely repeating the folder label
- analysis stays image-grounded and does not invent an unobservable root cause
- distractors are plausible but clearly wrong

If any item fails, rewrite or drop that QA instead of forcing an answer.

## Pilot expansion after smoke approval

Recommended controlled pilot:

- 52 vari-grip images
  - normal: 26
  - bird-nest: 13
  - rust: 13
- 5 QA per image
- total: 260 QA

The 26/13/13 split gives a 1:1 normal-vs-anomalous image balance and a 1:1 bird-nest-vs-rust balance within anomalous images.

The 260-QA pilot should only be generated after the smoke questions and answer policy pass manual visual review.
