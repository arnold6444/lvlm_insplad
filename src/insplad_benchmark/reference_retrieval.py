from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from .dataset import leakage_safe_train_normal_candidates

DEFAULT_MODEL_ID = 'facebook/dinov2-base'
DEFAULT_MODEL_REVISION = 'f9e44c814b77203eaa57a6bdbbd535f21ede1415'
DEFAULT_EMBEDDING_KIND = 'last_hidden_state_cls_l2'
RETRIEVAL_METHOD = 'dinov2_cosine_top1'
CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EmbeddingProvenance:
    model_id: str = DEFAULT_MODEL_ID
    model_revision: str | None = None
    embedding_kind: str = DEFAULT_EMBEDDING_KIND


class ImageEncoder(Protocol):
    @property
    def provenance(self) -> EmbeddingProvenance: ...

    def encode(self, images: Sequence[object]) -> np.ndarray: ...


def _atomic_replace(source: Path, target: Path, *, attempts: int = 5) -> None:
    """Retry transient Windows file locks while retaining atomic replace."""

    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.05 * (2 ** attempt))


class DinoV2Encoder:
    """Lazy Hugging Face DINOv2 encoder using the normalized CLS token."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        model_revision: str | None = None,
        device: str | None = None,
    ) -> None:
        if model_revision is None and model_id == DEFAULT_MODEL_ID:
            model_revision = DEFAULT_MODEL_REVISION
        self._provenance = EmbeddingProvenance(
            model_id=model_id,
            model_revision=model_revision,
        )
        self._requested_device = device
        self._device = None
        self._processor = None
        self._model = None
        self._torch = None

    @property
    def provenance(self) -> EmbeddingProvenance:
        return self._provenance

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError(
                'DINOv2 retrieval requires torch and transformers. '
                'Install them in the retrieval runtime before encoding images.'
            ) from exc

        kwargs = {}
        if self._provenance.model_revision:
            kwargs['revision'] = self._provenance.model_revision
        self._processor = AutoImageProcessor.from_pretrained(
            self._provenance.model_id,
            **kwargs,
        )
        self._model = AutoModel.from_pretrained(
            self._provenance.model_id,
            **kwargs,
        )
        device = self._requested_device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._device = torch.device(device)
        self._model.to(self._device)
        self._model.eval()
        self._torch = torch

    def encode(self, images: Sequence[object]) -> np.ndarray:
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        self._ensure_loaded()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None

        inputs = self._processor(images=list(images), return_tensors='pt')
        moved = {
            key: value.to(self._device) if hasattr(value, 'to') else value
            for key, value in inputs.items()
        }
        with self._torch.inference_mode():
            outputs = self._model(**moved)
            embeddings = outputs.last_hidden_state[:, 0, :].float()
            embeddings = self._torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().astype(np.float32, copy=False)

    def close(self) -> None:
        self._processor = None
        self._model = None
        self._torch = None
        self._device = None


@dataclass
class _CacheRecord:
    asset_category: str
    content_fingerprint: str
    embedding: np.ndarray


def _row_path(row: Mapping[str, object]) -> str:
    path = str(row.get('relative_path') or row.get('image') or '').strip()
    if not path:
        raise ValueError('embedding row is missing relative_path/image')
    return path


def _row_asset(row: Mapping[str, object]) -> str:
    asset = str(row.get('asset_category') or '').strip()
    if not asset:
        raise ValueError(f'{_row_path(row)}: embedding row is missing asset_category')
    return asset


def _row_fingerprint(row: Mapping[str, object]) -> str:
    fingerprint = str(row.get('content_fingerprint') or '').strip()
    if not fingerprint:
        raise ValueError(f'{_row_path(row)}: embedding row is missing content_fingerprint')
    return fingerprint


def _normalized_matrix(values: object, *, label: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError(f'{label}: expected a non-empty 2D embedding matrix')
    if not np.isfinite(matrix).all():
        raise ValueError(f'{label}: embeddings contain NaN or infinity')
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f'{label}: zero-norm embedding is not allowed')
    return matrix / norms


class NpzEmbeddingCache:
    """Pickle-free embedding cache that checkpoints every inserted batch."""

    def __init__(self, path: Path, provenance: EmbeddingProvenance) -> None:
        self.path = Path(path)
        self.provenance = provenance
        self._records: dict[str, _CacheRecord] = {}
        self._loaded = False
        self._dirty = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.is_file():
            return

        required = {
            'schema_version',
            'model_id',
            'model_revision',
            'embedding_kind',
            'paths',
            'asset_categories',
            'content_fingerprints',
            'embeddings',
        }
        with np.load(self.path, allow_pickle=False) as data:
            missing = required - set(data.files)
            if missing:
                raise ValueError(f'embedding cache is missing arrays: {sorted(missing)}')
            schema_version = int(data['schema_version'].item())
            stored = EmbeddingProvenance(
                model_id=str(data['model_id'].item()),
                model_revision=str(data['model_revision'].item()) or None,
                embedding_kind=str(data['embedding_kind'].item()),
            )
            if schema_version != CACHE_SCHEMA_VERSION:
                raise ValueError(
                    f'embedding cache schema mismatch: {schema_version} != {CACHE_SCHEMA_VERSION}'
                )
            if stored != self.provenance:
                raise ValueError(
                    'embedding cache provenance mismatch: '
                    f'stored={stored}, requested={self.provenance}'
                )

            paths = np.asarray(data['paths']).astype(str).tolist()
            assets = np.asarray(data['asset_categories']).astype(str).tolist()
            fingerprints = np.asarray(data['content_fingerprints']).astype(str).tolist()
            embeddings = np.asarray(data['embeddings'], dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError('embedding cache embeddings must be a 2D array')
        if not (len(paths) == len(assets) == len(fingerprints) == embeddings.shape[0]):
            raise ValueError('embedding cache arrays have inconsistent lengths')
        if len(paths) != len(set(paths)):
            raise ValueError('embedding cache contains duplicate paths')
        if embeddings.size and not np.isfinite(embeddings).all():
            raise ValueError('embedding cache contains NaN or infinity')

        for index, path in enumerate(paths):
            self._records[path] = _CacheRecord(
                asset_category=assets[index],
                content_fingerprint=fingerprints[index],
                embedding=embeddings[index],
            )

    def get(self, row: Mapping[str, object]) -> np.ndarray | None:
        self._load()
        record = self._records.get(_row_path(row))
        if record is None:
            return None
        if record.asset_category != _row_asset(row):
            return None
        if record.content_fingerprint != _row_fingerprint(row):
            return None
        return record.embedding

    def put_many(
        self,
        rows: Sequence[Mapping[str, object]],
        embeddings: object,
        *,
        flush: bool = True,
    ) -> None:
        self._load()
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[0] != len(rows) or matrix.shape[1] == 0:
            raise ValueError('cache update rows and embeddings have incompatible shapes')
        if not np.isfinite(matrix).all():
            raise ValueError('cache update embeddings contain NaN or infinity')

        existing_dimensions = {
            int(record.embedding.shape[0]) for record in self._records.values()
        }
        if existing_dimensions and existing_dimensions != {matrix.shape[1]}:
            raise ValueError(
                f'embedding cache dimension mismatch: {sorted(existing_dimensions)} != {matrix.shape[1]}'
            )
        for row, embedding in zip(rows, matrix):
            self._records[_row_path(row)] = _CacheRecord(
                asset_category=_row_asset(row),
                content_fingerprint=_row_fingerprint(row),
                embedding=embedding.copy(),
            )
        self._dirty = True
        if flush:
            self.flush()

    def flush(self) -> None:
        self._load()
        if self._dirty:
            self._write()

    def _write(self) -> None:
        paths = sorted(self._records)
        records = [self._records[path] for path in paths]
        embeddings = np.stack([record.embedding for record in records]).astype(
            np.float32,
            copy=False,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(self.path.name + '.tmp')
        with temp.open('wb') as handle:
            np.savez_compressed(
                handle,
                schema_version=np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int64),
                model_id=np.asarray(self.provenance.model_id, dtype=np.str_),
                model_revision=np.asarray(self.provenance.model_revision or '', dtype=np.str_),
                embedding_kind=np.asarray(self.provenance.embedding_kind, dtype=np.str_),
                paths=np.asarray(paths, dtype=np.str_),
                asset_categories=np.asarray(
                    [record.asset_category for record in records],
                    dtype=np.str_,
                ),
                content_fingerprints=np.asarray(
                    [record.content_fingerprint for record in records],
                    dtype=np.str_,
                ),
                embeddings=embeddings,
            )
        _atomic_replace(temp, self.path)
        self._dirty = False


def _unique_embedding_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    unique: dict[str, Mapping[str, object]] = {}
    for row in rows:
        path = _row_path(row)
        if path in unique:
            previous = unique[path]
            if (
                _row_asset(previous) != _row_asset(row)
                or _row_fingerprint(previous) != _row_fingerprint(row)
            ):
                raise ValueError(f'conflicting metadata for embedding path: {path}')
            continue
        unique[path] = row
    return [unique[path] for path in sorted(unique)]


def embed_rows_with_cache(
    rows: Sequence[Mapping[str, object]],
    *,
    encoder: ImageEncoder,
    image_loader: Callable[[Mapping[str, object]], object],
    cache_path: Path,
    batch_size: int = 16,
    checkpoint_every_batches: int = 25,
) -> dict[str, np.ndarray]:
    """Encode cache misses and atomically checkpoint bounded batches."""

    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if checkpoint_every_batches <= 0:
        raise ValueError('checkpoint_every_batches must be positive')
    ordered = _unique_embedding_rows(rows)
    cache = NpzEmbeddingCache(cache_path, encoder.provenance)
    result: dict[str, np.ndarray] = {}
    missing = []
    for row in ordered:
        cached = cache.get(row)
        if cached is None:
            missing.append(row)
        else:
            result[_row_path(row)] = cached

    batch_count = (len(missing) + batch_size - 1) // batch_size
    for batch_index, offset in enumerate(range(0, len(missing), batch_size), 1):
        batch = missing[offset:offset + batch_size]
        images = []
        try:
            for row in batch:
                images.append(image_loader(row))
            encoded = np.asarray(encoder.encode(images), dtype=np.float32)
        finally:
            for image in images:
                close = getattr(image, 'close', None)
                if callable(close):
                    close()
        if encoded.ndim == 1:
            encoded = encoded.reshape(1, -1)
        if encoded.ndim != 2 or encoded.shape[0] != len(batch):
            raise ValueError(
                f'encoder returned shape {encoded.shape} for batch of {len(batch)} images'
            )
        checkpoint = (
            batch_index % checkpoint_every_batches == 0
            or batch_index == batch_count
        )
        cache.put_many(batch, encoded, flush=checkpoint)
        for row, embedding in zip(batch, encoded):
            result[_row_path(row)] = embedding.copy()
    return result


def _manifest_index(
    unsupervised_rows: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    index: dict[str, Mapping[str, object]] = {}
    for row in unsupervised_rows:
        path = _row_path(row)
        if path in index:
            raise ValueError(f'duplicate manifest path: {path}')
        index[path] = row
    return index


def _embedding_matrix(
    paths: Sequence[str],
    embeddings: Mapping[str, object],
    *,
    label: str,
) -> np.ndarray:
    missing = [path for path in paths if path not in embeddings]
    if missing:
        raise ValueError(f'{label}: missing embeddings for {missing[:5]}')
    vectors = []
    for path in paths:
        vector = np.asarray(embeddings[path], dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError(f'{label}: embedding for {path} must be one-dimensional')
        vectors.append(vector)
    try:
        matrix = np.stack(vectors)
    except ValueError as exc:
        raise ValueError(f'{label}: embedding dimensions do not match') from exc
    return _normalized_matrix(matrix, label=label)


def retrieve_top1_references(
    query_rows: Sequence[Mapping[str, object]],
    unsupervised_rows: Sequence[Mapping[str, object]],
    embeddings: Mapping[str, object],
    *,
    batch_size: int = 256,
    provenance: EmbeddingProvenance | None = None,
    cache_sha256: str | None = None,
) -> list[dict]:
    """Retrieve deterministic same-asset Top-1 references by cosine similarity."""

    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    provenance = provenance or EmbeddingProvenance()
    manifest = _manifest_index(unsupervised_rows)
    assets = sorted({str(row.get('asset_category') or '') for row in query_rows})
    if not assets or '' in assets:
        raise ValueError('every query row must have asset_category')
    candidates_by_asset = leakage_safe_train_normal_candidates(
        unsupervised_rows,
        assets,
    )

    grouped_queries: dict[str, list[tuple[int, Mapping[str, object]]]] = {
        asset: [] for asset in assets
    }
    seen_query_ids = set()
    for index, row in enumerate(query_rows):
        query_id = str(row.get('id') or '').strip()
        if not query_id:
            raise ValueError(f'query row {index + 1} is missing id')
        if query_id in seen_query_ids:
            raise ValueError(f'duplicate query id: {query_id}')
        seen_query_ids.add(query_id)
        query_path = str(row.get('image') or '').strip()
        query_meta = manifest.get(query_path)
        if query_meta is None:
            raise ValueError(f'{query_id}: query image not found in manifest: {query_path}')
        asset = str(row['asset_category'])
        if str(query_meta.get('asset_category')) != asset:
            raise ValueError(f'{query_id}: query asset does not match manifest')
        if str(query_meta.get('split')).lower() != 'test':
            raise ValueError(f'{query_id}: query image is not from the test split')
        grouped_queries[asset].append((index, row))

    matches_by_index: dict[int, dict] = {}
    for asset in assets:
        candidates = candidates_by_asset[asset]
        candidate_paths = [str(row['relative_path']) for row in candidates]
        candidate_matrix = _embedding_matrix(
            candidate_paths,
            embeddings,
            label=f'{asset} candidates',
        )
        asset_queries = grouped_queries[asset]
        for offset in range(0, len(asset_queries), batch_size):
            batch = asset_queries[offset:offset + batch_size]
            query_paths = [str(row['image']) for _, row in batch]
            query_matrix = _embedding_matrix(
                query_paths,
                embeddings,
                label=f'{asset} queries',
            )
            if query_matrix.shape[1] != candidate_matrix.shape[1]:
                raise ValueError(f'{asset}: query and candidate embedding dimensions differ')
            scores = query_matrix @ candidate_matrix.T
            for batch_index, (original_index, query) in enumerate(batch):
                winner_index = int(np.argmax(scores[batch_index]))
                reference = candidates[winner_index]
                query_path = str(query['image'])
                query_meta = manifest[query_path]
                matches_by_index[original_index] = {
                    'query_id': str(query['id']),
                    'asset_category': asset,
                    'query_image': query_path,
                    'old_reference_image': query.get('reference_image'),
                    'retrieved_reference_image': reference['relative_path'],
                    'cosine_similarity': float(scores[batch_index, winner_index]),
                    'reference_split': reference['split'],
                    'reference_source_label_raw': reference['source_label_raw'],
                    'retrieval_method': RETRIEVAL_METHOD,
                    'retrieval_model_id': provenance.model_id,
                    'retrieval_model_revision': provenance.model_revision,
                    'retrieval_embedding': provenance.embedding_kind,
                    'retrieval_cache_sha256': cache_sha256,
                    'query_content_fingerprint': query_meta['content_fingerprint'],
                    'reference_content_fingerprint': reference['content_fingerprint'],
                }
    return [matches_by_index[index] for index in range(len(query_rows))]


def build_one_shot_plus_rows(
    base_one_shot_rows: Sequence[Mapping[str, object]],
    matches: Sequence[Mapping[str, object]],
) -> list[dict]:
    """Clone fixed one-shot QA while replacing only reference provenance."""

    match_by_id: dict[str, Mapping[str, object]] = {}
    for match in matches:
        query_id = str(match.get('query_id') or '').strip()
        if not query_id:
            raise ValueError('retrieval match is missing query_id')
        if query_id in match_by_id:
            raise ValueError(f'duplicate retrieval match for query id: {query_id}')
        match_by_id[query_id] = match

    base_ids = [str(row.get('id') or '') for row in base_one_shot_rows]
    if set(base_ids) != set(match_by_id):
        missing = sorted(set(base_ids) - set(match_by_id))
        extra = sorted(set(match_by_id) - set(base_ids))
        raise ValueError(f'retrieval match/query mismatch: missing={missing[:5]}, extra={extra[:5]}')

    output = []
    for base in base_one_shot_rows:
        query_id = str(base['id'])
        match = match_by_id[query_id]
        if str(match.get('query_image')) != str(base.get('image')):
            raise ValueError(f'{query_id}: retrieval match query image differs from QA')
        if str(match.get('asset_category')) != str(base.get('asset_category')):
            raise ValueError(f'{query_id}: retrieval match asset differs from QA')
        row = dict(base)
        row.update({
            'reference_image': match['retrieved_reference_image'],
            'reference_split': match['reference_split'],
            'reference_source_label_raw': match['reference_source_label_raw'],
            'retrieval_method': match['retrieval_method'],
            'reference_similarity': float(match['cosine_similarity']),
            'retrieval_model_id': match['retrieval_model_id'],
            'retrieval_model_revision': match.get('retrieval_model_revision'),
            'retrieval_embedding': match['retrieval_embedding'],
        })
        if match.get('retrieval_cache_sha256'):
            row['retrieval_cache_sha256'] = match['retrieval_cache_sha256']
        output.append(row)
    return output


def _finite_similarity(value: object, *, label: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{label}: similarity is not numeric') from exc
    if not math.isfinite(score):
        raise ValueError(f'{label}: similarity is not finite')
    if score < -1.000001 or score > 1.000001:
        raise ValueError(f'{label}: cosine similarity is outside [-1, 1]')
    return score


def validate_retrieval_provenance(
    base_one_shot_rows: Sequence[Mapping[str, object]],
    one_shot_plus_rows: Sequence[Mapping[str, object]],
    matches: Sequence[Mapping[str, object]],
    unsupervised_rows: Sequence[Mapping[str, object]],
    *,
    expected_method: str = RETRIEVAL_METHOD,
    expected_model_id: str | None = None,
    expected_model_revision: str | None = None,
    expected_embedding: str = DEFAULT_EMBEDDING_KIND,
    expected_cache_sha256: str | None = None,
) -> None:
    """Fail if IDs, GT, retrieval metadata, or leakage policy diverge."""

    errors: list[str] = []
    base_ids = [str(row.get('id') or '') for row in base_one_shot_rows]
    plus_ids = [str(row.get('id') or '') for row in one_shot_plus_rows]
    if base_ids != plus_ids:
        errors.append('base one-shot and one-shot+ query ID order differs')

    plus_by_id = {str(row.get('id') or ''): row for row in one_shot_plus_rows}
    match_by_id = {str(row.get('query_id') or ''): row for row in matches}
    if len(plus_by_id) != len(one_shot_plus_rows):
        errors.append('one-shot+ rows contain duplicate IDs')
    if len(match_by_id) != len(matches):
        errors.append('retrieval matches contain duplicate query IDs')
    if set(base_ids) != set(match_by_id):
        errors.append('retrieval matches do not cover exactly the base query IDs')

    try:
        manifest = _manifest_index(unsupervised_rows)
        assets = {str(row.get('asset_category') or '') for row in base_one_shot_rows}
        eligible = leakage_safe_train_normal_candidates(unsupervised_rows, assets)
        eligible_paths = {
            asset: {str(row['relative_path']) for row in rows}
            for asset, rows in eligible.items()
        }
    except ValueError as exc:
        errors.append(str(exc))
        manifest = {}
        eligible_paths = {}

    preserved_fields = ('id', 'image', 'asset_category', 'answer', 'task')
    for base in base_one_shot_rows:
        query_id = str(base.get('id') or '')
        plus = plus_by_id.get(query_id)
        match = match_by_id.get(query_id)
        if plus is None or match is None:
            continue
        for field in preserved_fields:
            if plus.get(field) != base.get(field):
                errors.append(f'{query_id}: preserved field changed: {field}')

        query_path = str(base.get('image') or '')
        asset = str(base.get('asset_category') or '')
        query_meta = manifest.get(query_path)
        if query_meta is None:
            errors.append(f'{query_id}: query image missing from manifest')
        else:
            if str(query_meta.get('asset_category')) != asset:
                errors.append(f'{query_id}: query manifest asset mismatch')
            if str(query_meta.get('split')).lower() != 'test':
                errors.append(f'{query_id}: query is not from test split')
            if query_meta.get('label') != base.get('answer'):
                errors.append(f'{query_id}: query GT differs from manifest')

        reference_path = str(plus.get('reference_image') or '')
        reference_meta = manifest.get(reference_path)
        if reference_path not in eligible_paths.get(asset, set()):
            errors.append(f'{query_id}: selected reference is not leakage-safe train-normal')
        if reference_meta is None:
            errors.append(f'{query_id}: selected reference missing from manifest')
        else:
            if str(reference_meta.get('asset_category')) != asset:
                errors.append(f'{query_id}: selected reference asset mismatch')
            if str(reference_meta.get('split')).lower() != 'train':
                errors.append(f'{query_id}: selected reference is not from train split')
            if reference_meta.get('label') != 'normal':
                errors.append(f'{query_id}: selected reference is not normal')
            if query_meta and reference_meta.get('content_fingerprint') == query_meta.get('content_fingerprint'):
                errors.append(f'{query_id}: duplicate query/reference fingerprint leakage')
            if plus.get('reference_split') != reference_meta.get('split'):
                errors.append(f'{query_id}: declared reference_split mismatch')
            if plus.get('reference_source_label_raw') != reference_meta.get('source_label_raw'):
                errors.append(f'{query_id}: declared reference source label mismatch')

        if plus.get('retrieval_method') != expected_method:
            errors.append(f'{query_id}: unexpected retrieval_method')
        if match.get('retrieval_method') != expected_method:
            errors.append(f'{query_id}: match retrieval_method mismatch')
        if expected_model_id is not None and plus.get('retrieval_model_id') != expected_model_id:
            errors.append(f'{query_id}: unexpected retrieval_model_id')
        if not str(plus.get('retrieval_model_revision') or '').strip():
            errors.append(f'{query_id}: missing immutable retrieval_model_revision')
        elif (
            expected_model_revision is not None
            and plus.get('retrieval_model_revision') != expected_model_revision
        ):
            errors.append(f'{query_id}: unexpected retrieval_model_revision')
        if plus.get('retrieval_embedding') != expected_embedding:
            errors.append(f'{query_id}: unexpected retrieval_embedding')
        cache_sha256 = str(plus.get('retrieval_cache_sha256') or '')
        if cache_sha256 and (
            len(cache_sha256) != 64
            or any(character not in '0123456789abcdefABCDEF' for character in cache_sha256)
        ):
            errors.append(f'{query_id}: invalid retrieval_cache_sha256')
        if expected_cache_sha256 is not None and cache_sha256 != expected_cache_sha256:
            errors.append(f'{query_id}: unexpected retrieval_cache_sha256')

        try:
            plus_score = _finite_similarity(
                plus.get('reference_similarity'),
                label=query_id,
            )
            match_score = _finite_similarity(
                match.get('cosine_similarity'),
                label=query_id,
            )
            if not math.isclose(plus_score, match_score, rel_tol=0.0, abs_tol=1e-9):
                errors.append(f'{query_id}: QA/match similarity differs')
        except ValueError as exc:
            errors.append(str(exc))

        expected_pairs = (
            ('query_image', 'image'),
            ('asset_category', 'asset_category'),
            ('retrieved_reference_image', 'reference_image'),
            ('reference_split', 'reference_split'),
            ('reference_source_label_raw', 'reference_source_label_raw'),
            ('retrieval_model_id', 'retrieval_model_id'),
            ('retrieval_model_revision', 'retrieval_model_revision'),
            ('retrieval_embedding', 'retrieval_embedding'),
            ('retrieval_cache_sha256', 'retrieval_cache_sha256'),
        )
        for match_field, plus_field in expected_pairs:
            if match.get(match_field) != plus.get(plus_field):
                errors.append(f'{query_id}: match/QA provenance differs: {match_field}')
        if match.get('old_reference_image') != base.get('reference_image'):
            errors.append(f'{query_id}: old fixed reference provenance differs')

    if errors:
        rendered = '; '.join(errors[:25])
        if len(errors) > 25:
            rendered += f'; ... {len(errors) - 25} more errors'
        raise ValueError(f'invalid one-shot+ retrieval provenance: {rendered}')
