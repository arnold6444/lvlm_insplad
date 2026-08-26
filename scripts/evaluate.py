#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from insplad_benchmark.metrics import (
    anomaly_classification_summary,
    balanced_accuracy,
    confusion_matrix,
    consistency_error,
    invalid_output_rate,
    macro_f1,
    per_class_prf,
)
from insplad_benchmark.parsing import parse_anomaly_output, parse_structured_output

INVALID = '<invalid>'


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


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
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def upsert_csv(path: Path, rows: list[dict], keys: tuple[str, ...]) -> None:
    old = []
    if path.exists() and path.stat().st_size:
        with path.open(newline='', encoding='utf-8') as f:
            old = list(csv.DictReader(f))
    new_keys = {tuple(str(r.get(k, '')) for k in keys) for r in rows}
    old = [r for r in old if tuple(str(r.get(k, '')) for k in keys) not in new_keys]
    write_csv(path, old + rows)


def asset_macro_summary(rows: list[dict], regime: str) -> dict:
    return {
        'regime': regime,
        'asset_n': len(rows),
        'balanced_accuracy': sum(float(row['balanced_accuracy']) for row in rows) / len(rows) if rows else 0.0,
        'f1': sum(float(row['f1']) for row in rows) / len(rows) if rows else 0.0,
    }


def eval_anomaly(gt: list[dict], predictions: dict[str, str], out_dir: Path, regime: str) -> None:
    def summary(rows: list[dict]) -> dict:
        parsed = [parse_anomaly_output(predictions.get(r['id'], '')) for r in rows]
        y_true = [r['answer'] for r in rows]
        y_pred = [p.anomaly if p.valid and p.anomaly else INVALID for p in parsed]
        return {'regime': regime, **anomaly_classification_summary(y_true, y_pred)}

    upsert_csv(out_dir / 'anomaly_overall.csv', [summary(gt)], ('regime',))
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in gt:
        groups[row['asset_category']].append(row)
    by_asset = []
    for asset, rows in sorted(groups.items()):
        data = summary(rows)
        data['asset_category'] = asset
        by_asset.append(data)
    upsert_csv(out_dir / 'anomaly_by_asset.csv', by_asset, ('regime', 'asset_category'))
    upsert_csv(
        out_dir / 'anomaly_asset_macro.csv',
        [asset_macro_summary(by_asset, regime)],
        ('regime',),
    )


def eval_explainability(gt: list[dict], predictions: dict[str, str], out_dir: Path) -> None:
    parsed = [
        parse_structured_output(predictions.get(r['id'], ''), set(r['valid_defect_classes']))
        for r in gt
    ]

    def summary(indices: list[int]) -> dict:
        anomaly_true = [gt[i]['answer']['anomaly'] for i in indices]
        defect_true = [gt[i]['answer']['defect_class'] for i in indices]
        anomaly_pred = [parsed[i].anomaly if parsed[i].valid else INVALID for i in indices]
        defect_pred = [parsed[i].defect_class if parsed[i].valid else INVALID for i in indices]
        labels = sorted(set(defect_true))
        joint = (
            sum(
                anomaly_true[j] == anomaly_pred[j] and defect_true[j] == defect_pred[j]
                for j in range(len(indices))
            ) / len(indices)
            if indices else 0.0
        )
        normal_indices = [j for j, a in enumerate(anomaly_true) if a == 'normal']
        false_defects = sum(
            parsed[indices[j]].valid and defect_pred[j] not in {'normal', INVALID}
            for j in normal_indices
        )
        pairs = [
            (anomaly_pred[j], defect_pred[j])
            for j in range(len(indices))
            if parsed[indices[j]].valid
        ]
        return {
            'n': len(indices),
            'valid_n': sum(parsed[i].valid for i in indices),
            'defect_balanced_accuracy': balanced_accuracy(defect_true, defect_pred, labels) if defect_true else 0.0,
            'defect_macro_f1': macro_f1(defect_true, defect_pred, labels) if defect_true else 0.0,
            'joint_accuracy': joint,
            'false_defect_rate': false_defects / len(normal_indices) if normal_indices else 0.0,
            'consistency_error': consistency_error(
                [a for a, _ in pairs], [d for _, d in pairs]
            ) if pairs else 0.0,
            'invalid_output_rate': invalid_output_rate([parsed[i].valid for i in indices]),
        }

    all_indices = list(range(len(gt)))
    write_csv(out_dir / 'explainability_overall.csv', [summary(all_indices)])

    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(gt):
        groups[row['asset_category']].append(i)
    by_asset = []
    for asset, indices in sorted(groups.items()):
        data = summary(indices)
        data['asset_category'] = asset
        by_asset.append(data)
    write_csv(out_dir / 'explainability_by_asset.csv', by_asset)

    defect_true = [r['answer']['defect_class'] for r in gt]
    defect_pred = [p.defect_class if p.valid else INVALID for p in parsed]
    labels = sorted(set(defect_true))
    write_csv(
        out_dir / 'explainability_by_defect.csv',
        [{'defect_class': key, **value} for key, value in per_class_prf(defect_true, defect_pred, labels).items()],
    )

    columns = labels + ([INVALID] if INVALID in defect_pred else [])
    matrix = confusion_matrix(defect_true, defect_pred, columns)
    rows = []
    for truth in labels:
        row = {'actual_defect': truth}
        row.update({f'pred_{pred}': matrix.get(truth, {}).get(pred, 0) for pred in columns})
        rows.append(row)
    write_csv(out_dir / 'explainability_confusion_matrix.csv', rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['anomaly', 'explainability'], required=True)
    parser.add_argument('--qa', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, default=Path('results'))
    parser.add_argument('--regime', choices=['zero_shot', 'one_shot', 'one_shot_plus'], default='zero_shot')
    args = parser.parse_args()
    gt = read_jsonl(args.qa)
    predictions = {r['id']: str(r.get('output', '')) for r in read_jsonl(args.predictions)}
    if args.task == 'anomaly':
        eval_anomaly(gt, predictions, args.out_dir, args.regime)
    else:
        eval_explainability(gt, predictions, args.out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
