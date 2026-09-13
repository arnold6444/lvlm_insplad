#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

APPROVE = {'approve', 'approved', 'keep', 'pass', 'ok'}
EXCLUDE = {'exclude', 'excluded', 'drop', 'reject', 'remove'}


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def normalize_status(value: object) -> str:
    return str(value or '').strip().lower()


def newest(pattern: str, root: Path) -> Path | None:
    values = sorted(root.glob(pattern))
    return values[-1] if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description='Freeze gold-v2 after every candidate received an explicit quality review decision.')
    parser.add_argument('--candidate-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()

    review_path = args.candidate_dir / 'manual_review.csv'
    if not review_path.is_file():
        raise FileNotFoundError(review_path)
    reviews = read_csv(review_path)
    if not reviews:
        raise RuntimeError('manual_review.csv is empty')

    approved: set[str] = set()
    excluded: set[str] = set()
    pending: list[str] = []
    invalid: list[tuple[str, str]] = []
    for row in reviews:
        core_id = str(row.get('core_id', ''))
        status = normalize_status(row.get('quality_status'))
        if status in APPROVE:
            approved.add(core_id)
        elif status in EXCLUDE:
            excluded.add(core_id)
        elif not status:
            pending.append(core_id)
        else:
            invalid.append((core_id, status))
    if invalid:
        raise RuntimeError(f'invalid quality_status values, first={invalid[:10]}')
    if pending:
        raise RuntimeError(f'{len(pending)} candidates still pending review; first={pending[:10]}')
    if not approved:
        raise RuntimeError('no approved candidates')

    master_path = newest('master_images_*.jsonl', args.candidate_dir)
    qa_path = newest('master_qa_*.jsonl', args.candidate_dir)
    if master_path is None or qa_path is None:
        raise FileNotFoundError('candidate master_images/master_qa files not found')
    master = read_jsonl(master_path)
    qa = read_jsonl(qa_path)
    final_master = [dict(row, quality_review_status='approved') for row in master if str(row['core_id']) in approved]
    final_qa = [dict(row, quality_review_status='approved', review_status='gold') for row in qa if str(row['core_id']) in approved]

    loc_images = read_jsonl(args.candidate_dir / 'localization_images.jsonl')
    loc_qa_path = newest('localization_qa_*.jsonl', args.candidate_dir)
    loc_qa = read_jsonl(loc_qa_path) if loc_qa_path else []
    final_loc_images = [row for row in loc_images if str(row['core_id']) in approved]
    final_loc_qa = [dict(row, quality_review_status='approved', review_status='gold') for row in loc_qa if str(row['core_id']) in approved]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / f'master_images_{len(final_master)}.jsonl', final_master)
    write_jsonl(args.out_dir / f'master_qa_{len(final_qa)}.jsonl', final_qa)
    if final_loc_images:
        write_jsonl(args.out_dir / 'localization_images.jsonl', final_loc_images)
    if final_loc_qa:
        write_jsonl(args.out_dir / f'localization_qa_{len(final_loc_qa)}.jsonl', final_loc_qa)

    counts = []
    for asset in sorted({row['asset_category'] for row in final_master}):
        rows = [row for row in final_master if row['asset_category'] == asset]
        counts.append({
            'asset_category': asset,
            'total': len(rows),
            'normal': sum(row['anomaly_gt'] == 'normal' for row in rows),
            'anomalous': sum(row['anomaly_gt'] == 'anomalous' for row in rows),
        })
    write_csv(args.out_dir / 'master_counts.csv', counts)
    write_csv(args.out_dir / 'manual_review_final.csv', reviews)

    summary = {
        'candidate_count': len(reviews),
        'approved_count': len(approved),
        'excluded_count': len(excluded),
        'final_master_images': len(final_master),
        'final_q1_q2_questions': len(final_qa),
        'final_localization_images': len(final_loc_images),
        'final_localization_questions': len(final_loc_qa),
        'sampling_policy': 'all anomalies + median-capped DINOv2-diverse normals, followed by complete manual quality review',
        'quality_status': 'FROZEN',
    }
    (args.out_dir / 'build_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # Best-effort publication to MyDrive root for local/Drive discoverability.
    try:
        from publish_latest_results import publish
        published = publish(
            args.out_dir,
            'lvlm_gold_v2_final',
            ('build_summary.json', 'master_counts.csv', 'manual_review_final.csv'),
        )
        print(json.dumps({'drive_publish': published}, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f'[drive-publish skipped] {type(exc).__name__}: {exc}')

    # Best-effort publication of small summaries to GitHub so ChatGPT can read
    # results immediately without waiting for Google Drive indexing.
    if os.environ.get('GITHUB_TOKEN', '').strip():
        try:
            from publish_results_to_github import publish as publish_github
            published = publish_github(
                args.out_dir,
                repo='arnold6444/lvlm_insplad',
                branch='main',
                prefix='gold_v2_final',
                token=os.environ['GITHUB_TOKEN'],
                filenames=('build_summary.json', 'master_counts.csv', 'manual_review_final.csv'),
            )
            print(json.dumps({'github_publish': published}, ensure_ascii=False, indent=2))
        except Exception as exc:
            print(f'[github-publish skipped] {type(exc).__name__}: {exc}')
    else:
        print('[github-publish skipped] GITHUB_TOKEN is not set in environment')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
