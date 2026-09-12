#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_gold_baselines as gold


def stable_key(seed: int, text: str) -> str:
    return hashlib.sha256(f'{seed}:{text}'.encode()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8'); return
    fields=[]
    for row in rows:
        for k in row:
            if k not in fields: fields.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def choose(rows: list[dict], per_asset: int, seed: int) -> list[dict]:
    by_asset=defaultdict(list)
    for row in rows: by_asset[row['asset_category']].append(row)
    selected=[]
    for asset in sorted(by_asset):
        normals=sorted([r for r in by_asset[asset] if r['anomaly_gt']=='normal'], key=lambda r: stable_key(seed, r['core_id']))
        anomalies=sorted([r for r in by_asset[asset] if r['anomaly_gt']=='anomalous'], key=lambda r: stable_key(seed, r['core_id']))
        target_normal=min(len(normals), per_asset//2)
        target_anom=min(len(anomalies), per_asset-target_normal)
        bucket=normals[:target_normal]+anomalies[:target_anom]
        remain=per_asset-len(bucket)
        if remain:
            used={r['core_id'] for r in bucket}
            extra=sorted([r for r in by_asset[asset] if r['core_id'] not in used], key=lambda r: stable_key(seed+1, r['core_id']))
            bucket += extra[:remain]
        if len(bucket) != per_asset:
            raise RuntimeError(f'{asset}: cannot select {per_asset}, only {len(bucket)}')
        selected.extend(bucket)
    return sorted(selected, key=lambda r: stable_key(seed+2, r['core_id']))


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument('--qa-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--per-asset', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    a=p.parse_args()
    rows=gold.load_gold_qa(a.qa_dir)
    by_core={}
    for r in rows:
        by_core.setdefault(r['core_id'], r)
    images=[{
        # build_vision_expert_smoke_data.py requires a stable, unique `id` field.
        # Use core_id so ROI manifests can be joined back to all Q1/Q2/Q3 rows.
        'id': r['core_id'],
        'core_id': r['core_id'], 'asset_category': r['asset_category'],
        'image': r['unsupervised_image'], 'supervised_image': r['image'],
        'reference_image': r['normal_reference_image'],
        'answer': r['anomaly_gt'], 'anomaly_gt': r['anomaly_gt'], 'defect_gt': r['defect_gt'],
    } for r in by_core.values()]
    chosen=choose(images, a.per_asset, a.seed)
    ids={r['core_id'] for r in chosen}
    mcqa=[dict(r) for r in rows if r['core_id'] in ids]
    mcqa.sort(key=lambda r:(r['core_id'], gold.QUESTION_ORDER.get(r['question_type'],99)))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(a.out_dir/'expert_smoke_images.jsonl', chosen)
    write_jsonl(a.out_dir/'expert_smoke_mcqa.jsonl', mcqa)
    summary=[]
    for asset in sorted({r['asset_category'] for r in chosen}):
        ar=[r for r in chosen if r['asset_category']==asset]
        summary.append({'asset_category':asset,'images':len(ar),'normal':sum(r['answer']=='normal' for r in ar),'anomalous':sum(r['answer']=='anomalous' for r in ar),'mcqa':sum(r['asset_category']==asset for r in mcqa),'q3':sum(r['asset_category']==asset and r['question_type']=='defect_localization' for r in mcqa)})
    write_csv(a.out_dir/'selection_summary.csv', summary)
    meta={'seed':a.seed,'per_asset':a.per_asset,'image_count':len(chosen),'mcqa_count':len(mcqa),'q_counts':dict(Counter(r['question_type'] for r in mcqa)),'core_ids':sorted(ids)}
    (a.out_dir/'selection_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
