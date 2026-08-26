#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from insplad_benchmark.dataset import parse_supervised_manifest, parse_unsupervised_manifest
from insplad_benchmark.validation import read_jsonl, validate_aligned_anomaly_sets, validate_qa_rows


def build_index(anomaly_manifest: Path | None, fault_manifest: Path | None):
    index, assets, defects = {}, set(), set()
    if anomaly_manifest:
        for row in parse_unsupervised_manifest(anomaly_manifest):
            index[row['relative_path']] = row; assets.add(row['asset_category'])
    if fault_manifest:
        rows, _ = parse_supervised_manifest(fault_manifest)
        for row in rows:
            meta = dict(row); meta['label'] = 'normal' if row['defect_class'] == 'normal' else 'anomalous'
            index[row['relative_path']] = meta; assets.add(row['asset_category']); defects.add(row['defect_class'])
    return index, assets, defects


def main() -> int:
    p = argparse.ArgumentParser(description='Validate InsPLAD Benchmark QA JSONL.')
    p.add_argument('qa', nargs='*', type=Path)
    p.add_argument('--anomaly-manifest', type=Path); p.add_argument('--fault-manifest', type=Path)
    p.add_argument('--report', type=Path, default=Path('reports/qa_validation.md')); args = p.parse_args()
    qa_paths = args.qa or [Path('data/qa/anomaly_zero_shot.jsonl'), Path('data/qa/anomaly_one_shot.jsonl'), Path('data/qa/defect_qa.jsonl')]
    index, assets, defects = build_index(args.anomaly_manifest, args.fault_manifest)
    loaded, results, ok = {}, [], True
    for path in qa_paths:
        try: rows = read_jsonl(path)
        except (ValueError, FileNotFoundError) as exc:
            results.append((path, None, [str(exc)])); ok = False; continue
        loaded[path.name] = rows
        result = validate_qa_rows(rows, assets or None, defects or None, index or None)
        results.append((path, result, [])); ok = ok and result.ok
    alignment = []
    zero_rows = loaded.get('anomaly_zero_shot.jsonl')
    if zero_rows is None:
        zero_rows = loaded.get('zero_shot.jsonl')
    fixed_rows = loaded.get('anomaly_one_shot.jsonl')
    if fixed_rows is None:
        fixed_rows = loaded.get('one_shot.jsonl')
    plus_rows = loaded.get('one_shot_plus.jsonl')
    if zero_rows is not None and fixed_rows is not None:
        errors = validate_aligned_anomaly_sets(zero_rows, fixed_rows)
        alignment.append(('zero-shot / fixed one-shot', errors)); ok = ok and not errors
    if zero_rows is not None and plus_rows is not None:
        errors = validate_aligned_anomaly_sets(zero_rows, plus_rows)
        alignment.append(('zero-shot / retrieved 1-shot+', errors)); ok = ok and not errors
    lines = ['# QA Validation', '', f"- Status: {'PASS' if ok else 'FAIL'}", '']
    for path, result, errors in results:
        lines += [f'## `{path}`', '']
        if errors: lines += ['- Status: FAIL', *[f'- {e}' for e in errors], '']; continue
        lines += [f"- Status: {'PASS' if result.ok else 'FAIL'}", f'- Rows checked: {result.checked}', f'- Errors: {len(result.errors)}', f'- Warnings: {len(result.warnings)}', '']
        if result.errors: lines += ['### Errors', *[f'- {e}' for e in result.errors[:100]], '']
    lines += ['## Anomaly QA alignment', '']
    if not alignment:
        lines += ['- Not checked: no comparable anomaly QA pairs were loaded.', '']
    else:
        for name, errors in alignment:
            lines.append(f"- {'PASS' if not errors else 'FAIL'} `{name}`: " + ('identical IDs, images, and GT answers' if not errors else '; '.join(errors)))
        lines.append('')
    lines += ['## Validation scope', '', '- Manifest-backed image/category/split/label checks are enabled when manifests are provided.', '- Every reference must be a same-asset train normal.', '- Duplicate content uses SHA256 if available, otherwise size+ZIP CRC32 fingerprint.', '- Exact cross-dataset SHA256 overlap is confirmed separately with `scripts/hash_overlap.py`.', '']
    args.report.parent.mkdir(parents=True, exist_ok=True); args.report.write_text('\n'.join(lines), encoding='utf-8')
    return 0 if ok else 1

if __name__ == '__main__': raise SystemExit(main())
