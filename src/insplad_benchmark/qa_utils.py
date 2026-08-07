from __future__ import annotations

import random
from typing import Iterable, Mapping, Sequence

from .dataset import normalize_key


def normalize_label(value: str) -> str:
    return normalize_key(value).replace('-', '_')


def category_intersection(left: Iterable[str], right: Iterable[str]) -> list[str]:
    return sorted(set(left) & set(right))


def anomaly_from_defect(defect_class: str, normal_label: str = 'normal') -> str:
    normals = {normal_label, 'good', 'normal'}
    return 'normal' if normalize_key(defect_class) in {normalize_key(x) for x in normals} else 'anomalous'


def choose_normal_reference(test_sample: Mapping[str, str], candidates: Sequence[Mapping[str, str]], *, seed: int = 42, normal_label: str = 'normal') -> Mapping[str, str]:
    eligible = []
    test_path = test_sample.get('relative_path', '')
    test_hash = test_sample.get('sha256', '')
    test_fp = test_sample.get('content_fingerprint', '')
    for candidate in candidates:
        if candidate.get('asset_category') != test_sample.get('asset_category'): continue
        if anomaly_from_defect(str(candidate.get('label', '')), normal_label) != 'normal': continue
        if str(candidate.get('split', '')).lower() == 'test': continue
        if candidate.get('relative_path') == test_path: continue
        if test_hash and candidate.get('sha256') == test_hash: continue
        if test_fp and candidate.get('content_fingerprint') == test_fp: continue
        eligible.append(candidate)
    if not eligible:
        raise ValueError(f"no eligible normal reference for asset {test_sample.get('asset_category')!r}")
    eligible.sort(key=lambda row: str(row.get('relative_path', '')))
    return random.Random(seed).choice(eligible)
