#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from insplad_benchmark.dataset import (
    build_anomaly_qa, build_defect_qa, category_intersection, cross_split_duplicate_candidates,
    defect_classes_by_asset, exact_overlap_candidates, normalize_defect, normalize_key,
    parse_supervised_manifest, parse_unsupervised_manifest, select_reference_by_asset, stratified_sample,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_csv(path: Path, rows: list[dict]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8'); return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def label_counts(unsup: list[dict], sup: list[dict]) -> list[dict]:
    counts = Counter()
    for r in unsup:
        counts[('unsupervised_anomaly_detection', r['asset_category'], r['split'], r['source_label_raw'], r['label'])] += 1
    for r in sup:
        counts[('supervised_fault_classification', r['asset_category'], r['split'], r['source_defect_raw'], r['defect_class'])] += 1
    return [{'dataset': d, 'asset_category': a, 'split': s, 'raw_label': raw, 'normalized_label': norm, 'count': n} for (d,a,s,raw,norm),n in sorted(counts.items())]


def structure_report(unsup: list[dict], sup: list[dict], annotations: list[dict], common: list[str], eval_split: str, reason: str) -> str:
    u = defaultdict(lambda: defaultdict(set)); s = defaultdict(lambda: defaultdict(set))
    for r in unsup: u[r['asset_category']][r['split']].add(r['source_label_raw'])
    for r in sup: s[r['asset_category']][r['split']].add(r['source_defect_raw'])
    lines = ['# InsPLAD fault dataset structure (manifest-verified)', '', '## unsupervised_anomaly_detection.zip', '', '```text', 'unsupervised_anomaly_detection/']
    for asset in sorted(u):
        lines.append(f'├─ {asset}/')
        for split in sorted(u[asset]): lines.append(f"│  ├─ {split}/  [{', '.join(sorted(u[asset][split]))}]")
    lines += ['```', '', f'- Asset categories: **{len(u)}**', '- Splits: `train`, `test`', '- Images: `.jpg`', '', '## supervised_fault_classification.zip', '', '```text', 'defect_supervised/']
    for asset in sorted(s):
        lines.append(f'├─ {asset}/')
        for split in sorted(s[asset]): lines.append(f"│  ├─ {split}/  [{', '.join(sorted(s[asset][split]))}]")
        lines.append('│  └─ annotations/image_list_val.txt')
    lines += ['```', '', f'- Asset categories: **{len(s)}**', '- Splits: `train`, `val` (no `test` observed)', f'- Annotation-like files: **{len(annotations)}** `.txt` files', '- Annotation contents are not inferred from the manifest.', '', '## Benchmark Core', '', *[f'- `{x}`' for x in common], '', f'- Structured evaluation split: `{eval_split}` ({reason})', '- Anomaly evaluation: official unsupervised `test`, restricted to these common categories.', '']
    return '\n'.join(lines)


def sample_review(sections: list[tuple[str, list[dict]]], seed: int) -> str:
    lines = ['# QA sample review', '', f'Deterministic stratified review sample (`seed={seed}`).', '', 'Inspect the listed local source image and compare it with the recorded GT. One-shot rows also list the fixed normal reference.', '']
    for title, rows in sections:
        lines += [f'## {title}', '']
        for i, row in enumerate(rows, 1):
            lines.append(f"{i}. `{row['id']}` — **{row['asset_category']}**")
            lines.append(f"   - image: `{row['image']}`")
            if row.get('reference_image'): lines.append(f"   - reference: `{row['reference_image']}`")
            lines.append(f"   - GT: `{json.dumps(row['answer'], ensure_ascii=False)}`")
        lines.append('')
    return '\n'.join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--anomaly-manifest', type=Path, required=True)
    p.add_argument('--fault-manifest', type=Path, required=True)
    p.add_argument('--repo-root', type=Path, default=ROOT)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(); root = args.repo_root
    unsup = parse_unsupervised_manifest(args.anomaly_manifest)
    sup, annotations = parse_supervised_manifest(args.fault_manifest)
    common = category_intersection(unsup, sup)
    refs = select_reference_by_asset(unsup, common, seed=args.seed)
    zero = build_anomaly_qa(unsup, common)
    one = build_anomaly_qa(unsup, common, references=refs)
    defect, eval_split, reason = build_defect_qa(sup)
    overlap = exact_overlap_candidates(sup, unsup)

    write_jsonl(root/'data/qa/anomaly_zero_shot.jsonl', zero)
    write_jsonl(root/'data/qa/anomaly_one_shot.jsonl', one)
    write_jsonl(root/'data/qa/defect_qa.jsonl', defect)
    zero_sample = stratified_sample(zero, 20, seed=args.seed)
    one_by_id = {r['id']: r for r in one}
    one_sample = [one_by_id[r['id']] for r in zero_sample]
    defect_sample = stratified_sample(defect, 20, seed=args.seed)
    write_jsonl(root/'samples/anomaly_zero_shot_20.jsonl', zero_sample)
    write_jsonl(root/'samples/anomaly_one_shot_20.jsonl', one_sample)
    write_jsonl(root/'samples/defect_20.jsonl', defect_sample)
    write_csv(root/'reports/category_counts.csv', label_counts(unsup, sup))
    write_csv(root/'reports/common_categories.csv', [{'asset_category': a, 'in_unsupervised': True, 'in_supervised': True} for a in common])
    write_csv(root/'reports/image_overlap.csv', overlap)
    (root/'reports/dataset_structure.md').write_text(structure_report(unsup, sup, annotations, common, eval_split, reason), encoding='utf-8')
    (root/'reports/sample_review.md').write_text(sample_review([('Anomaly zero-shot', zero_sample), ('Anomaly one-shot', one_sample), ('Structured explainability', defect_sample)], args.seed), encoding='utf-8')

    u_leak = cross_split_duplicate_candidates(unsup); s_leak = cross_split_duplicate_candidates(sup)
    issues = ['# Data issues and benchmark decisions', '', f'- Supervised has no test split; structured evaluation uses `{eval_split}`.', '- Source normal labels `good` and `normal` are both normalized to benchmark `normal`.', '- `corrosão` is mechanically normalized to `corrosao` and is NOT merged with `rust`.', f'- Cross-dataset overlap candidates: **{len(overlap)}** by filename+size+ZIP CRC32; confirm with `scripts/hash_overlap.py` for SHA256.', f'- Unsupervised cross-split duplicate candidates: **{len(u_leak)}**.', f'- Supervised cross-split duplicate candidates: **{len(s_leak)}**.', f'- Annotation-like files: **{len(annotations)}**; manifest does not include their contents.', '', 'One-shot references use one same-asset train normal per asset, seed 42, excluding train candidates whose size+CRC32 fingerprint occurs in test.']
    if u_leak:
        issues += ['', '## Unsupervised train/test duplicate candidate']
        for x in u_leak: issues.append('- ' + ' | '.join(f"`{p}`" for p in x['paths']))
    (root/'reports/data_issues.md').write_text('\n'.join(issues)+'\n', encoding='utf-8')

    (root/'configs').mkdir(parents=True, exist_ok=True)
    category_map = {'raw_to_normalized': {x: normalize_key(x) for x in sorted({r['asset_category_raw'] for r in unsup}|{r['asset_category_raw'] for r in sup})}, 'benchmark_core': common}
    defect_map = {'raw_to_normalized': {x: normalize_defect(x) for x in sorted({r['source_defect_raw'] for r in sup})}, 'valid_by_asset': defect_classes_by_asset(sup), 'normal_source_labels': sorted({r['source_defect_raw'] for r in sup if r['defect_class']=='normal'})}
    (root/'configs/category_mapping.json').write_text(json.dumps(category_map, ensure_ascii=False, indent=2), encoding='utf-8')
    (root/'configs/defect_mapping.json').write_text(json.dumps(defect_map, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {'unsupervised_assets': sorted({r['asset_category'] for r in unsup}), 'supervised_assets': sorted({r['asset_category'] for r in sup}), 'common_assets': common, 'anomaly_qa_count': len(zero), 'anomaly_one_shot_count': len(one), 'defect_qa_count': len(defect), 'structured_eval_split': eval_split, 'overlap_candidate_count': len(overlap), 'references': {a:r['relative_path'] for a,r in refs.items()}}
    (root/'reports/build_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0

if __name__ == '__main__': raise SystemExit(main())
