from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class ValidationResult:
    checked: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip(): continue
            try: value = json.loads(line)
            except json.JSONDecodeError as exc: raise ValueError(f'{path}:{line_no}: invalid JSON: {exc.msg}') from exc
            if not isinstance(value, dict): raise ValueError(f'{path}:{line_no}: JSONL row must be an object')
            rows.append(value)
    return rows


def _same_content(a: dict | None, b: dict | None) -> bool:
    if not a or not b: return False
    if a.get('sha256') and b.get('sha256'): return a['sha256'] == b['sha256']
    if a.get('content_fingerprint') and b.get('content_fingerprint'): return a['content_fingerprint'] == b['content_fingerprint']
    return False


def validate_qa_rows(rows: Iterable[dict], known_assets: set[str] | None = None, known_defects: set[str] | None = None, image_meta: dict[str, dict] | None = None) -> ValidationResult:
    result = ValidationResult(); seen_ids = set(); image_meta = image_meta or {}
    test_fingerprints = {
        str(meta.get('content_fingerprint'))
        for meta in image_meta.values()
        if str(meta.get('split', '')).lower() == 'test' and meta.get('content_fingerprint')
    }
    for row in rows:
        result.checked += 1
        row_id = str(row.get('id', '')).strip(); label = row_id or f'row {result.checked}'
        if not row_id: result.errors.append(f'row {result.checked}: missing id')
        elif row_id in seen_ids: result.errors.append(f'duplicate id: {row_id}')
        seen_ids.add(row_id)
        image = str(row.get('image', '')).strip(); asset = str(row.get('asset_category', '')).strip()
        if not image: result.errors.append(f'{label}: missing image')
        if known_assets is not None and asset not in known_assets: result.errors.append(f'{label}: unknown asset: {asset}')
        meta = image_meta.get(image) if image_meta and image else None
        if image_meta and image and meta is None: result.errors.append(f'{label}: image not found in index: {image}')
        elif meta and asset and meta.get('asset_category') != asset: result.errors.append(f'{label}: image asset mismatch')

        task = row.get('task')
        if task == 'anomaly_detection':
            if row.get('answer') not in {'normal', 'anomalous'}: result.errors.append(f'{label}: invalid anomaly GT: {row.get("answer")}')
            if meta:
                if str(meta.get('split', '')).lower() != 'test': result.errors.append(f'{label}: anomaly query is not from test split')
                if meta.get('label') != row.get('answer'): result.errors.append(f'{label}: anomaly GT differs from manifest')
        elif task == 'structured_explainability':
            answer = row.get('answer') or {}; anomaly = answer.get('anomaly'); defect = answer.get('defect_class')
            if known_defects is not None and defect not in known_defects: result.errors.append(f'{label}: unknown defect: {defect}')
            if 'valid_defect_classes' in row and defect not in (row.get('valid_defect_classes') or []): result.errors.append(f'{label}: answer defect not in valid_defect_classes: {defect}')
            expected = 'normal' if defect == 'normal' else 'anomalous'
            if anomaly != expected: result.errors.append(f'{label}: GT inconsistency ({anomaly}, {defect})')
        else: result.errors.append(f'{label}: unknown task: {task}')

        reference = str(row.get('reference_image', '')).strip()
        if reference:
            if reference == image: result.errors.append(f'{label}: reference equals test image')
            ref_meta = image_meta.get(reference) if image_meta else None
            if image_meta and ref_meta is None: result.errors.append(f'{label}: reference not found in index: {reference}')
            if ref_meta:
                if ref_meta.get('label') != 'normal': result.errors.append(f'{label}: reference is not normal')
                if ref_meta.get('asset_category') != asset: result.errors.append(f'{label}: reference category mismatch')
                if str(ref_meta.get('split', '')).lower() != 'train': result.errors.append(f'{label}: reference is not from train split')
                if _same_content(ref_meta, meta): result.errors.append(f'{label}: duplicate content leakage')
                ref_fingerprint = str(ref_meta.get('content_fingerprint') or '')
                if ref_fingerprint and ref_fingerprint in test_fingerprints:
                    result.errors.append(f'{label}: reference fingerprint occurs in test split')
                declared_split = str(row.get('reference_split', '')).strip().lower()
                if declared_split and declared_split != str(ref_meta.get('split', '')).strip().lower():
                    result.errors.append(f'{label}: reference_split metadata mismatch')
                declared_label = str(row.get('reference_source_label_raw', '')).strip()
                source_label = str(ref_meta.get('source_label_raw', '')).strip()
                if declared_label and source_label and declared_label != source_label:
                    result.errors.append(f'{label}: reference_source_label_raw metadata mismatch')

        retrieval_method = str(row.get('retrieval_method', '')).strip()
        has_similarity = 'reference_similarity' in row
        if retrieval_method or has_similarity:
            if retrieval_method != 'dinov2_cosine_top1':
                result.errors.append(f'{label}: invalid retrieval_method: {retrieval_method or "<missing>"}')
            if not reference:
                result.errors.append(f'{label}: retrieved row is missing reference_image')
            if str(row.get('reference_split', '')).strip().lower() != 'train':
                result.errors.append(f'{label}: retrieved row reference_split must be train')
            if not str(row.get('reference_source_label_raw', '')).strip():
                result.errors.append(f'{label}: retrieved row is missing reference_source_label_raw')
            if not str(row.get('retrieval_model_id', '')).strip():
                result.errors.append(f'{label}: retrieved row is missing retrieval_model_id')
            if not str(row.get('retrieval_model_revision', '')).strip():
                result.errors.append(f'{label}: retrieved row is missing retrieval_model_revision')
            retrieval_embedding = str(row.get('retrieval_embedding', '')).strip()
            if not retrieval_embedding:
                result.errors.append(f'{label}: retrieved row is missing retrieval_embedding')
            elif retrieval_embedding != 'last_hidden_state_cls_l2':
                result.errors.append(f'{label}: invalid retrieval_embedding: {retrieval_embedding}')
            cache_sha256 = str(row.get('retrieval_cache_sha256', '')).strip()
            if len(cache_sha256) != 64 or any(
                character not in '0123456789abcdefABCDEF'
                for character in cache_sha256
            ):
                result.errors.append(f'{label}: invalid retrieval_cache_sha256')
            try:
                similarity = float(row.get('reference_similarity'))
            except (TypeError, ValueError):
                result.errors.append(f'{label}: invalid reference_similarity')
            else:
                if not math.isfinite(similarity) or similarity < -1.000001 or similarity > 1.000001:
                    result.errors.append(f'{label}: reference_similarity outside cosine range')
    return result


def validate_aligned_anomaly_sets(zero_rows: list[dict], one_rows: list[dict]) -> list[str]:
    fields = ('id', 'image', 'asset_category', 'answer')
    zero = [tuple(row.get(field) for field in fields) for row in zero_rows]
    one = [tuple(row.get(field) for field in fields) for row in one_rows]
    return [] if zero == one else [
        'zero-shot and one-shot anomaly rows are not identical in count/order/ID/image/asset/GT'
    ]
