#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_explainable_qa_gold as base


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def build_master_rows(selected_rows: list[dict[str, str]], seed: int) -> tuple[list[dict], list[dict]]:
    images: list[dict] = []
    questions: list[dict] = []
    for row in selected_rows:
        item = {
            'core_id': row['core_id'],
            'asset_category': row['asset_category'],
            'image': row['supervised_image'],
            'unsupervised_image': row['unsupervised_image'],
            'normal_reference_image': row['reference_image'],
            'anomaly_gt': row['anomaly_answer'],
            'defect_gt': row['defect_class'],
            'anomaly_gt_basis': 'verified_aligned_source_label',
            'defect_gt_basis': 'verified_supervised_source_label',
            'exact_sha256_match': True,
            'binary_gt_agreement': True,
            'sample_policy': 'all_anomalies_plus_median_capped_diverse_normals_then_manual_quality_review',
            'selection_reason': row.get('selection_reason', ''),
        }
        images.append(item)

        anomaly_text = 'Normal.' if item['anomaly_gt'] == 'normal' else 'Anomalous.'
        options, answer = base.make_options(
            ['Normal.', 'Anomalous.'], anomaly_text,
            seed_text=f"{seed}:{item['core_id']}:anomaly",
        )
        questions.append({
            **item,
            'id': f"{item['core_id']}-q1",
            'task': 'explainable_mcqa',
            'question_type': 'anomaly_detection',
            'question': 'Is the target power-line asset normal or anomalous?',
            'options': options,
            'answer': answer,
            'answer_text': anomaly_text,
            'gt_basis': 'verified_aligned_source_label',
            'review_status': 'candidate',
        })

        defect_text = base.DEFECT_TEXT[item['defect_gt']]
        options, answer = base.make_options(
            list(base.DEFECT_TEXT.values()), defect_text,
            seed_text=f"{seed}:{item['core_id']}:defect",
        )
        questions.append({
            **item,
            'id': f"{item['core_id']}-q2",
            'task': 'explainable_mcqa',
            'question_type': 'defect_classification',
            'question': 'Which condition best describes the target power-line asset?',
            'options': options,
            'answer': answer,
            'answer_text': defect_text,
            'gt_basis': 'verified_supervised_source_label',
            'review_status': 'candidate',
        })
    return images, questions


def main() -> int:
    parser = argparse.ArgumentParser(description='Build quality-reviewable InsPLAD gold-v2 QA from a preselected candidate set.')
    parser.add_argument('--aligned-core-csv', type=Path, required=True)
    parser.add_argument('--selection-csv', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seg-root', type=Path)
    parser.add_argument('--seg-version', type=int, default=7)
    args = parser.parse_args()

    aligned_rows = read_csv(args.aligned_core_csv)
    if len(aligned_rows) != 1378:
        raise RuntimeError(f'expected 1378 aligned-core rows, got {len(aligned_rows)}')
    selection = read_csv(args.selection_csv)
    selected_ids = {row['core_id'] for row in selection if str(row.get('selected', 'true')).strip().lower() in {'1','true','yes','y'}}
    if not selected_ids:
        selected_ids = {row['core_id'] for row in selection}
    aligned_by_id = {row['core_id']: row for row in aligned_rows}
    missing = sorted(selected_ids - set(aligned_by_id))
    if missing:
        raise RuntimeError(f'selection contains unknown core IDs: {missing[:10]}')
    selection_by_id = {row['core_id']: row for row in selection}
    selected_rows = []
    for core_id in sorted(selected_ids):
        row = dict(aligned_by_id[core_id])
        row['selection_reason'] = selection_by_id.get(core_id, {}).get('selection_reason', '')
        selected_rows.append(row)

    master_images, master_qa = build_master_rows(selected_rows, args.seed)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    base.write_jsonl(out / f'master_images_{len(master_images)}.jsonl', master_images)
    base.write_jsonl(out / f'master_qa_{len(master_qa)}.jsonl', master_qa)
    base.write_csv(out / 'master_counts.csv', base.count_rows(master_images))

    localization_images: list[dict] = []
    localization_qa: list[dict] = []
    audit: list[dict] = []
    if args.seg_root:
        localization_images, localization_qa, audit = base.build_localization_rows(
            aligned_rows, master_images, master_qa, args.seg_root,
            seed=args.seed, seg_version=args.seg_version,
        )
        for row in localization_qa:
            row['review_status'] = 'candidate'
        base.write_jsonl(out / 'localization_images.jsonl', localization_images)
        base.write_jsonl(out / f'localization_qa_{len(localization_qa)}.jsonl', localization_qa)
        base.write_csv(out / 'localization_match_audit.csv', audit)

    loc_by_core = {str(row['core_id']): row for row in localization_images}
    review_rows = []
    for row in master_images:
        core_id = str(row['core_id'])
        loc = loc_by_core.get(core_id)
        review_rows.append({
            'core_id': core_id,
            'asset_category': row['asset_category'],
            'image': row['image'],
            'anomaly_gt': row['anomaly_gt'],
            'defect_gt': row['defect_gt'],
            'selection_reason': selection_by_id.get(core_id, {}).get('selection_reason', ''),
            'has_localization_gt': bool(loc),
            'localization_region': loc['localization_gt']['coarse_region'] if loc else '',
            'quality_status': '',
            'quality_issue': '',
            'review_notes': '',
        })
    base.write_csv(out / 'manual_review.csv', review_rows)

    summary = {
        'aligned_core_images': len(aligned_rows),
        'candidate_images_before_quality_review': len(master_images),
        'candidate_questions_q1_q2': len(master_qa),
        'localization_images': len(localization_images),
        'localization_questions': len(localization_qa),
        'sampling_policy': 'all anomalies + per-asset normal median cap; over-cap normals selected by DINOv2 k-center; manual quality review before freeze',
        'quality_status': 'NOT_FROZEN',
        'seed': args.seed,
    }
    (out / 'build_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
