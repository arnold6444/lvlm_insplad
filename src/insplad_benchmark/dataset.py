from __future__ import annotations

import csv
import hashlib
import json
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


def all_assets(rows: Iterable[Mapping[str, object]]) -> list[str]:
    return sorted({str(r['asset_category']) for r in rows})


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
                'supervised_anomaly': s['anomaly'],
                'unsupervised_split': u['split'], 'unsupervised_label_raw': u['source_label_raw'],
                'unsupervised_anomaly': u['label'], 'filename': s['filename'], 'size': s['size'],
                'crc32': s['crc32'], 'match_basis': 'filename+size+zip_crc32',
            })
    return out


def load_overlap_confirmation(path: Path, candidate_count: int) -> dict:
    data = json.loads(path.read_text(encoding='utf-8'))
    required = {'candidate_count', 'sha256_match_count', 'sha256_mismatch_count'}
    missing = required - set(data)
    if missing:
        raise ValueError(f'overlap summary missing fields: {sorted(missing)}')
    if int(data['candidate_count']) != candidate_count:
        raise ValueError(f"overlap summary candidate_count={data['candidate_count']} does not match manifest candidates={candidate_count}")
    if int(data['sha256_match_count']) + int(data['sha256_mismatch_count']) != candidate_count:
        raise ValueError('overlap summary SHA256 counts do not add up to candidate_count')
    data['all_candidates_sha256_confirmed'] = (
        int(data['sha256_match_count']) == candidate_count and int(data['sha256_mismatch_count']) == 0
    )
    return data


def aligned_core_rows(overlap: Sequence[Mapping[str, object]], *, exact_sha256_confirmed: bool) -> tuple[list[dict], list[dict]]:
    if not exact_sha256_confirmed:
        return [], []
    aligned = [
        dict(r) for r in overlap
        if str(r.get('unsupervised_split')) == 'test' and str(r.get('supervised_split')) == 'val'
    ]
    clean, conflicts = [], []
    for row in aligned:
        row['exact_sha256_match'] = True
        if row.get('unsupervised_anomaly') == row.get('supervised_anomaly'):
            row['binary_gt_agreement'] = True
            row['core_anomaly'] = row['unsupervised_anomaly']
            clean.append(row)
        else:
            row['binary_gt_agreement'] = False
            conflicts.append(row)
    return clean, conflicts


def core_link_rows(clean_overlap: Sequence[Mapping[str, object]], references: Mapping[str, Mapping[str, object]]) -> list[dict]:
    rows = []
    for row in sorted(clean_overlap, key=lambda r: (str(r['asset_category']), str(r['unsupervised_path']))):
        asset = str(row['asset_category'])
        unsup_path = str(row['unsupervised_path'])
        sup_path = str(row['supervised_path'])
        rows.append({
            'core_id': stable_id('core', unsup_path),
            'asset_category': asset,
            'unsupervised_image': unsup_path,
            'supervised_image': sup_path,
            'anomaly_id': stable_id('anomaly', unsup_path),
            'structured_id': stable_id('defect', sup_path),
            'reference_image': references[asset]['relative_path'],
            'anomaly_answer': row['core_anomaly'],
            'defect_class': row['supervised_defect'],
            'unsupervised_source_label_raw': row['unsupervised_label_raw'],
            'supervised_source_defect_raw': row['supervised_defect_raw'],
            'exact_sha256_match': True,
            'binary_gt_agreement': True,
        })
    return rows


def parse_annotation_previews(path: Path) -> dict:
    values = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(values, list):
        raise ValueError('annotation preview JSON must be a list')
    by_asset: dict[str, dict] = {}
    for item in values:
        rel = str(item.get('path', ''))
        parts = PurePosixPath(rel).parts
        if len(parts) < 4:
            continue
        asset = normalize_key(parts[-3])
        class_ids: dict[str, set[int]] = defaultdict(set)
        parsed_lines = 0
        for line in str(item.get('preview', '')).splitlines():
            match = re.match(r'^(.*?)\s+(-?\d+)\s*$', line)
            if not match:
                continue
            image_rel, class_id = match.group(1), int(match.group(2))
            defect_raw = PurePosixPath(image_rel).parts[0] if PurePosixPath(image_rel).parts else ''
            if defect_raw:
                class_ids[defect_raw].add(class_id)
                parsed_lines += 1
        by_asset[asset] = {
            'annotation_path': rel,
            'truncated': bool(item.get('truncated')),
            'preview_lines_parsed': parsed_lines,
            'raw_defect_to_class_ids': {k: sorted(v) for k, v in sorted(class_ids.items())},
        }

    vari = by_asset.get('vari-grip', {}).get('raw_defect_to_class_ids', {})
    numeric_ids_are_binary_anomaly = not (
        1 in vari.get('good', []) and 0 in vari.get('bird-nest', [])
    )
    return {
        'line_schema': '<relative_image_path> <integer_class_id>',
        'by_asset': by_asset,
        'numeric_class_id_is_binary_anomaly': numeric_ids_are_binary_anomaly,
        'gt_policy': 'Use defect folder names for defect/anomaly GT; do not interpret annotation class IDs as binary anomaly labels.',
    }


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


def leakage_safe_train_normal_candidates(
    unsupervised_rows: Sequence[Mapping[str, object]],
    assets: Iterable[str],
) -> dict[str, list[Mapping[str, object]]]:
    assets = sorted(set(assets))
    # A content duplicate is leakage even when a mislabeled/cross-copied file
    # appears under another asset directory. Exclude every fingerprint present
    # anywhere in the unsupervised test split before grouping by asset.
    test_fingerprints = {
        str(row['content_fingerprint'])
        for row in unsupervised_rows
        if row['split'] == 'test'
    }
    by_asset = {}
    for asset in assets:
        candidates = [r for r in unsupervised_rows if r['asset_category'] == asset and r['split'] == 'train' and r['label'] == 'normal' and str(r['content_fingerprint']) not in test_fingerprints]
        if not candidates:
            raise ValueError(f'no leakage-safe train normal reference for asset {asset}')
        candidates.sort(key=lambda r: str(r['relative_path']))
        by_asset[asset] = candidates
    return by_asset


def select_reference_by_asset(unsupervised_rows: Sequence[Mapping[str, object]], assets: Iterable[str], *, seed: int = 42) -> dict[str, Mapping[str, object]]:
    candidates_by_asset = leakage_safe_train_normal_candidates(unsupervised_rows, assets)
    refs = {}
    for asset, candidates in candidates_by_asset.items():
        refs[asset] = random.Random(f'{seed}:{asset}').choice(candidates)
    return refs


def stable_id(prefix: str, relative_path: str) -> str:
    return f"{prefix}-{hashlib.sha256(relative_path.encode('utf-8')).hexdigest()[:16]}"


def build_anomaly_qa(unsupervised_rows: Sequence[Mapping[str, object]], assets: Iterable[str], *, references: Mapping[str, Mapping[str, object]] | None = None) -> list[dict]:
    selected_assets = set(assets)
    rows = sorted((r for r in unsupervised_rows if r['asset_category'] in selected_assets and r['split'] == 'test'), key=lambda r: str(r['relative_path']))
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
            values.sort(key=lambda r: str(r.get('id') or r.get('core_id') or r))
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


def stratified_core_sample(rows: Sequence[Mapping[str, object]], n: int = 20, *, seed: int = 42) -> list[dict]:
    decorated = []
    for row in rows:
        item = dict(row)
        item['id'] = item['core_id']
        item['answer'] = {'anomaly': item['anomaly_answer'], 'defect_class': item['defect_class']}
        decorated.append(item)
    sampled = stratified_sample(decorated, n=n, seed=seed)
    for row in sampled:
        row.pop('id', None)
        row.pop('answer', None)
    return sampled
