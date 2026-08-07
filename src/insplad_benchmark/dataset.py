from __future__ import annotations

import csv
import hashlib
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
NORMAL_SOURCE_LABELS = {'good', 'normal'}


def normalize_key(value: str) -> str:
    text = unicodedata.normalize('NFKD', value.strip()).encode('ascii', 'ignore').decode('ascii').lower()
    return re.sub(r'[^a-z0-9]+', '-', text).strip('-')


def normalize_defect(value: str) -> str:
    key = normalize_key(value)
    return 'normal' if key in NORMAL_SOURCE_LABELS else key


def anomaly_from_source_label(value: str) -> str:
    return 'normal' if normalize_key(value) in NORMAL_SOURCE_LABELS else 'anomalous'


def _as_bool(value: object) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in {'1', 'true', 'yes', 'y'}


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def _fingerprint(row: Mapping[str, object]) -> str:
    return f"{int(row.get('size', 0))}:{str(row.get('crc32', '')).lower()}"


def parse_unsupervised_manifest(path: Path) -> list[dict]:
    rows = []
    for raw in _read_manifest(path):
        if _as_bool(raw.get('is_dir')) or str(raw.get('extension') or '').lower() not in IMAGE_EXTENSIONS:
            continue
        parts = PurePosixPath(raw['path']).parts
        if len(parts) < 4:
            continue
        asset, split, source_label = parts[-4], parts[-3], parts[-2]
        row = {
            'dataset': 'unsupervised_anomaly_detection', 'relative_path': raw['path'],
            'asset_category_raw': asset, 'asset_category': normalize_key(asset), 'split': split.lower(),
            'source_label_raw': source_label, 'label': anomaly_from_source_label(source_label),
            'filename': parts[-1], 'size': int(raw.get('size') or 0), 'crc32': str(raw.get('crc32') or '').lower(),
        }
        row['content_fingerprint'] = _fingerprint(row)
        rows.append(row)
    return rows


def parse_supervised_manifest(path: Path) -> tuple[list[dict], list[dict]]:
    images, annotations = [], []
    for raw in _read_manifest(path):
        if _as_bool(raw.get('is_dir')):
            continue
        parts = PurePosixPath(raw['path']).parts
        ext = str(raw.get('extension') or '').lower()
        if ext in IMAGE_EXTENSIONS and len(parts) >= 5:
            asset, split, source_defect = parts[-4], parts[-3], parts[-2]
            defect = normalize_defect(source_defect)
            row = {
                'dataset': 'supervised_fault_classification', 'relative_path': raw['path'],
                'asset_category_raw': asset, 'asset_category': normalize_key(asset), 'split': split.lower(),
                'source_defect_raw': source_defect, 'defect_class': defect,
                'anomaly': 'normal' if defect == 'normal' else 'anomalous', 'filename': parts[-1],
                'size': int(raw.get('size') or 0), 'crc32': str(raw.get('crc32') or '').lower(),
            }
            row['content_fingerprint'] = _fingerprint(row)
            images.append(row)
        elif ext in {'.txt', '.csv', '.json', '.jsonl', '.xml', '.yaml', '.yml'}:
            annotations.append({'relative_path': raw['path'], 'extension': ext, 'size': int(raw.get('size') or 0), 'crc32': str(raw.get('crc32') or '').lower()})
    return images, annotations


def category_intersection(left: Iterable[Mapping[str, object]], right: Iterable[Mapping[str, object]]) -> list[str]:
    return sorted({str(r['asset_category']) for r in left} & {str(r['asset_category']) for r in right})


def choose_evaluation_split(rows: Sequence[Mapping[str, object]]) -> tuple[str, str]:
    splits = {str(r.get('split', '')).lower() for r in rows}
    if 'test' in splits:
        return 'test', 'official test split'
    if 'val' in splits:
        return 'val', 'no test split present; using official validation split'
    if 'validation' in splits:
        return 'validation', 'no test split present; using official validation split'
    raise ValueError(f'no supported evaluation split found: {sorted(splits)}')


def defect_classes_by_asset(rows: Sequence[Mapping[str, object]]) -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        grouped[str(row['asset_category'])].add(str(row['defect_class']))
    out = {}
    for asset, labels in sorted(grouped.items()):
        out[asset] = (['normal'] if 'normal' in labels else []) + sorted(labels - {'normal'})
    return out


def exact_overlap_candidates(supervised: Sequence[Mapping[str, object]], unsupervised: Sequence[Mapping[str, object]]) -> list[dict]:
    u_index: dict[tuple[str, int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in unsupervised:
        u_index[(str(row['filename']), int(row['size']), str(row['crc32']))].append(row)
    out = []
    for s in supervised:
        key = (str(s['filename']), int(s['size']), str(s['crc32']))
        for u in u_index.get(key, []):
            out.append({
                'supervised_path': s['relative_path'], 'unsupervised_path': u['relative_path'],
                'asset_category': s['asset_category'], 'supervised_split': s['split'],
                'supervised_defect_raw': s['source_defect_raw'], 'supervised_defect': s['defect_class'],
                'unsupervised_split': u['split'], 'unsupervised_label_raw': u['source_label_raw'],
                'unsupervised_anomaly': u['label'], 'filename': s['filename'], 'size': s['size'],
                'crc32': s['crc32'], 'match_basis': 'filename+size+zip_crc32',
            })
    return out


def cross_split_duplicate_candidates(rows: Sequence[Mapping[str, object]]) -> list[dict]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row['content_fingerprint'])].append(row)
    out = []
    for fingerprint, group in grouped.items():
        splits = sorted({str(r['split']) for r in group})
        if len(splits) > 1:
            out.append({'content_fingerprint': fingerprint, 'splits': splits, 'paths': [str(r['relative_path']) for r in group]})
    return out


def select_reference_by_asset(unsupervised_rows: Sequence[Mapping[str, object]], assets: Iterable[str], *, seed: int = 42) -> dict[str, Mapping[str, object]]:
    assets = sorted(set(assets))
    test_fingerprints: dict[str, set[str]] = defaultdict(set)
    for row in unsupervised_rows:
        if row['asset_category'] in assets and row['split'] == 'test':
            test_fingerprints[str(row['asset_category'])].add(str(row['content_fingerprint']))
    refs = {}
    for asset in assets:
        candidates = [r for r in unsupervised_rows if r['asset_category'] == asset and r['split'] == 'train' and r['label'] == 'normal' and str(r['content_fingerprint']) not in test_fingerprints[asset]]
        if not candidates:
            raise ValueError(f'no leakage-safe train normal reference for asset {asset}')
        candidates.sort(key=lambda r: str(r['relative_path']))
        refs[asset] = random.Random(f'{seed}:{asset}').choice(candidates)
    return refs


def stable_id(prefix: str, relative_path: str) -> str:
    return f"{prefix}-{hashlib.sha256(relative_path.encode('utf-8')).hexdigest()[:16]}"


def build_anomaly_qa(unsupervised_rows: Sequence[Mapping[str, object]], common_assets: Iterable[str], *, references: Mapping[str, Mapping[str, object]] | None = None) -> list[dict]:
    assets = set(common_assets)
    rows = sorted((r for r in unsupervised_rows if r['asset_category'] in assets and r['split'] == 'test'), key=lambda r: str(r['relative_path']))
    qa = []
    for row in rows:
        item = {
            'id': stable_id('anomaly', str(row['relative_path'])), 'task': 'anomaly_detection',
            'image': row['relative_path'], 'asset_category': row['asset_category'], 'asset_category_raw': row['asset_category_raw'],
            'split': row['split'], 'question': 'Inspect the power-line asset. Is it normal or anomalous?',
            'options': ['normal', 'anomalous'], 'answer': row['label'], 'source_label_raw': row['source_label_raw'],
        }
        if references is not None:
            ref = references[str(row['asset_category'])]
            item.update({'reference_image': ref['relative_path'], 'reference_split': ref['split'], 'reference_source_label_raw': ref['source_label_raw']})
        qa.append(item)
    return qa


def build_defect_qa(supervised_rows: Sequence[Mapping[str, object]]) -> tuple[list[dict], str, str]:
    eval_split, reason = choose_evaluation_split(supervised_rows)
    valid_map = defect_classes_by_asset(supervised_rows)
    rows = sorted((r for r in supervised_rows if r['split'] == eval_split), key=lambda r: str(r['relative_path']))
    qa = []
    for row in rows:
        qa.append({
            'id': stable_id('defect', str(row['relative_path'])), 'task': 'structured_explainability',
            'image': row['relative_path'], 'asset_category': row['asset_category'], 'asset_category_raw': row['asset_category_raw'],
            'split': row['split'], 'question': 'Determine whether the asset is normal or anomalous. If anomalous, identify the defect type.',
            'valid_defect_classes': valid_map[str(row['asset_category'])],
            'answer': {'anomaly': row['anomaly'], 'defect_class': row['defect_class']}, 'source_defect_raw': row['source_defect_raw'],
        })
    return qa, eval_split, reason


def stratified_sample(rows: Sequence[Mapping[str, object]], n: int = 20, *, seed: int = 42) -> list[dict]:
    if n >= len(rows):
        return [dict(r) for r in rows]
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row['asset_category'])].append(row)
    assets = sorted(grouped)
    base, extra = divmod(n, len(assets))
    rng = random.Random(seed)
    selected = []
    for i, asset in enumerate(assets):
        take = base + (1 if i < extra else 0)
        label_groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in grouped[asset]:
            answer = row.get('answer')
            label = str(answer.get('defect_class')) if isinstance(answer, dict) else str(answer)
            label_groups[label].append(row)
        for values in label_groups.values():
            values.sort(key=lambda r: str(r['id']))
            rng.shuffle(values)
        labels, cursor, pool = sorted(label_groups), 0, []
        while len(pool) < take and labels:
            for label in labels:
                if cursor < len(label_groups[label]):
                    pool.append(label_groups[label][cursor])
                    if len(pool) == take:
                        break
            cursor += 1
            labels = [label for label in labels if cursor < len(label_groups[label])]
        selected.extend(dict(r) for r in pool[:take])
    return selected[:n]
