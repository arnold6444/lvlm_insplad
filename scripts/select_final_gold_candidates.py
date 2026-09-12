#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from io import BytesIO
from pathlib import Path
from typing import Iterable, Mapping
from zipfile import ZipFile

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalized_member(value: str) -> str:
    return str(value).replace('\\', '/').lstrip('/')


def zip_member_index(zip_path: Path) -> dict[str, str]:
    with ZipFile(zip_path) as archive:
        index: dict[str, str] = {}
        for name in archive.namelist():
            key = normalized_member(name)
            if key in index and index[key] != name:
                raise RuntimeError(f'ambiguous ZIP member: {key}')
            index[key] = name
    return index


def load_images(zip_path: Path, members: list[str]):
    from PIL import Image

    index = zip_member_index(zip_path)
    images = []
    with ZipFile(zip_path) as archive:
        for member in members:
            key = normalized_member(member)
            actual = index.get(key)
            if actual is None:
                raise FileNotFoundError(f'ZIP member not found: {member}')
            with Image.open(BytesIO(archive.read(actual))) as image:
                images.append(image.convert('RGB').copy())
    return images


def embed_rows(rows: list[dict[str, str]], zip_path: Path, model_id: str, batch_size: int, device: str) -> np.ndarray:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    vectors: list[np.ndarray] = []
    members = [str(row['supervised_image']) for row in rows]

    for start in range(0, len(rows), batch_size):
        batch_members = members[start:start + batch_size]
        images = load_images(zip_path, batch_members)
        inputs = processor(images=images, return_tensors='pt')
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model(**inputs)
            hidden = output.last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden, dim=1)
        vectors.append(hidden.detach().cpu().float().numpy())
        print(f'[dino] {min(start + batch_size, len(rows))}/{len(rows)}', flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(vectors, axis=0)


def kcenter_indices(features: np.ndarray, k: int, ids: list[str]) -> list[int]:
    if k >= len(features):
        return list(range(len(features)))
    if k <= 0:
        return []
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.clip(norms, 1e-12, None)

    centroid = features.mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
    centroid_distance = 1.0 - features @ centroid
    first = min(range(len(features)), key=lambda i: (float(centroid_distance[i]), ids[i]))
    selected = [first]
    min_distance = 1.0 - features @ features[first]
    min_distance[first] = -1.0

    while len(selected) < k:
        max_value = float(np.max(min_distance))
        candidates = [i for i, value in enumerate(min_distance) if math.isclose(float(value), max_value, rel_tol=0.0, abs_tol=1e-10)]
        next_index = min(candidates, key=lambda i: ids[i])
        selected.append(next_index)
        distance = 1.0 - features @ features[next_index]
        min_distance = np.minimum(min_distance, distance)
        min_distance[selected] = -1.0
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description='Select final gold-v2 candidates with all anomalies + median-capped diverse normals.')
    parser.add_argument('--aligned-core-csv', type=Path, required=True)
    parser.add_argument('--supervised-zip', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--model-id', default='facebook/dinov2-small')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    rows = read_csv(args.aligned_core_csv)
    if len(rows) != 1378:
        raise RuntimeError(f'expected 1378 aligned-core rows, got {len(rows)}')
    assets = sorted({row['asset_category'] for row in rows})
    normals_by_asset = {
        asset: [row for row in rows if row['asset_category'] == asset and row['anomaly_answer'] == 'normal']
        for asset in assets
    }
    anomalies = [row for row in rows if row['anomaly_answer'] == 'anomalous']
    normal_counts = [len(normals_by_asset[asset]) for asset in assets]
    cap = int(statistics.median(normal_counts))
    if cap <= 0:
        raise RuntimeError(f'invalid median normal cap: {cap}')

    if args.device == 'auto':
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            device = 'cpu'
    else:
        device = args.device

    selected_ids = {row['core_id'] for row in anomalies}
    selection_reason = {row['core_id']: 'all_valid_anomalies' for row in anomalies}
    selection_rank: dict[str, int | str] = {row['core_id']: '' for row in anomalies}

    for asset in assets:
        normals = sorted(normals_by_asset[asset], key=lambda row: row['core_id'])
        if len(normals) <= cap:
            chosen = list(range(len(normals)))
            reason = 'all_normals_below_median_cap'
        else:
            print(f'[{asset}] normal={len(normals)} > cap={cap}; selecting DINOv2-diverse representatives', flush=True)
            features = embed_rows(normals, args.supervised_zip, args.model_id, args.batch_size, device)
            chosen = kcenter_indices(features, cap, [row['core_id'] for row in normals])
            reason = 'dinov2_kcenter_normal_representatives'
        for rank, index in enumerate(chosen, start=1):
            core_id = normals[index]['core_id']
            selected_ids.add(core_id)
            selection_reason[core_id] = reason
            selection_rank[core_id] = rank

    audit_rows = []
    selected_rows = []
    for row in rows:
        selected = row['core_id'] in selected_ids
        out = {
            **row,
            'selected': selected,
            'selection_reason': selection_reason.get(row['core_id'], 'normal_over_cap_not_selected'),
            'selection_rank_within_asset': selection_rank.get(row['core_id'], ''),
            'normal_cap_policy': 'median_normal_count_across_assets',
            'normal_cap': cap,
            'normal_selection_method': 'DINOv2 CLS embedding + deterministic k-center greedy when over cap',
        }
        audit_rows.append(out)
        if selected:
            selected_rows.append(out)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / 'selection_audit_all_1378.csv', audit_rows)
    write_csv(args.out_dir / 'selected_candidates.csv', selected_rows)
    write_csv(
        args.out_dir / 'quality_review.csv',
        [
            {
                'core_id': row['core_id'],
                'asset_category': row['asset_category'],
                'anomaly_gt': row['anomaly_answer'],
                'defect_gt': row['defect_class'],
                'supervised_image': row['supervised_image'],
                'selection_reason': row['selection_reason'],
                'quality_status': '',
                'quality_issue': '',
                'review_notes': '',
            }
            for row in selected_rows
        ],
    )

    summary_rows = []
    for asset in assets:
        asset_selected = [row for row in selected_rows if row['asset_category'] == asset]
        summary_rows.append({
            'asset_category': asset,
            'source_total': sum(row['asset_category'] == asset for row in rows),
            'source_normal': len(normals_by_asset[asset]),
            'source_anomalous': sum(row['asset_category'] == asset and row['anomaly_answer'] == 'anomalous' for row in rows),
            'selected_total_before_quality_review': len(asset_selected),
            'selected_normal': sum(row['anomaly_answer'] == 'normal' for row in asset_selected),
            'selected_anomalous': sum(row['anomaly_answer'] == 'anomalous' for row in asset_selected),
            'normal_cap': cap,
        })
    write_csv(args.out_dir / 'selection_summary.csv', summary_rows)
    metadata = {
        'aligned_core_count': len(rows),
        'all_anomalies_preserved': len(anomalies),
        'normal_counts_by_asset': {asset: len(normals_by_asset[asset]) for asset in assets},
        'median_normal_cap': cap,
        'candidate_count_before_quality_review': len(selected_rows),
        'normal_feature_model': args.model_id,
        'normal_selection': 'all normals when <= cap; otherwise deterministic DINOv2 k-center representatives',
        'quality_policy': 'manual pre-model review; exclude only unreadable/ambiguous/duplicate-like samples before any final benchmark inference',
    }
    (args.out_dir / 'selection_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
