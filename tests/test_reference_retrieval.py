from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import insplad_benchmark.reference_retrieval as retrieval
from insplad_benchmark.reference_retrieval import (
    DEFAULT_MODEL_REVISION,
    DinoV2Encoder,
    EmbeddingProvenance,
    NpzEmbeddingCache,
    RETRIEVAL_METHOD,
    build_one_shot_plus_rows,
    embed_rows_with_cache,
    retrieve_top1_references,
    validate_retrieval_provenance,
)


def _manifest_rows():
    return [
        {
            'relative_path': 'a/test/good/q1.jpg',
            'asset_category': 'a',
            'split': 'test',
            'label': 'normal',
            'source_label_raw': 'good',
            'content_fingerprint': '10:q1',
        },
        {
            'relative_path': 'a/test/rust/q2.jpg',
            'asset_category': 'a',
            'split': 'test',
            'label': 'anomalous',
            'source_label_raw': 'rust',
            'content_fingerprint': '11:q2',
        },
        {
            'relative_path': 'a/train/good/ref-b.jpg',
            'asset_category': 'a',
            'split': 'train',
            'label': 'normal',
            'source_label_raw': 'good',
            'content_fingerprint': '12:b',
        },
        {
            'relative_path': 'a/train/good/ref-a.jpg',
            'asset_category': 'a',
            'split': 'train',
            'label': 'normal',
            'source_label_raw': 'good',
            'content_fingerprint': '13:a',
        },
    ]


def _base_rows():
    return [
        {
            'id': 'q1',
            'task': 'anomaly_detection',
            'image': 'a/test/good/q1.jpg',
            'asset_category': 'a',
            'answer': 'normal',
            'reference_image': 'a/train/good/ref-a.jpg',
            'reference_split': 'train',
            'reference_source_label_raw': 'good',
        },
        {
            'id': 'q2',
            'task': 'anomaly_detection',
            'image': 'a/test/rust/q2.jpg',
            'asset_category': 'a',
            'answer': 'anomalous',
            'reference_image': 'a/train/good/ref-a.jpg',
            'reference_split': 'train',
            'reference_source_label_raw': 'good',
        },
    ]


def _embeddings():
    return {
        'a/test/good/q1.jpg': np.asarray([1.0, 0.0], dtype=np.float32),
        'a/test/rust/q2.jpg': np.asarray([0.0, 1.0], dtype=np.float32),
        'a/train/good/ref-a.jpg': np.asarray([1.0, 0.0], dtype=np.float32),
        'a/train/good/ref-b.jpg': np.asarray([0.0, 1.0], dtype=np.float32),
    }


def test_batched_cosine_top1_builds_query_specific_qa_and_provenance():
    base = _base_rows()
    manifest = _manifest_rows()
    provenance = EmbeddingProvenance(
        model_id='fake/dinov2',
        model_revision='revision-1',
    )

    matches = retrieve_top1_references(
        base,
        manifest,
        _embeddings(),
        batch_size=1,
        provenance=provenance,
        cache_sha256='a' * 64,
    )
    plus = build_one_shot_plus_rows(base, matches)
    validate_retrieval_provenance(
        base,
        plus,
        matches,
        manifest,
        expected_model_id='fake/dinov2',
        expected_cache_sha256='a' * 64,
    )

    assert [row['retrieved_reference_image'] for row in matches] == [
        'a/train/good/ref-a.jpg',
        'a/train/good/ref-b.jpg',
    ]
    assert [row['reference_image'] for row in plus] == [
        'a/train/good/ref-a.jpg',
        'a/train/good/ref-b.jpg',
    ]
    assert [(row['id'], row['image'], row['answer']) for row in plus] == [
        (row['id'], row['image'], row['answer']) for row in base
    ]
    assert all(row['retrieval_method'] == RETRIEVAL_METHOD for row in plus)
    assert all(row['retrieval_cache_sha256'] == 'a' * 64 for row in plus)
    assert all(row['reference_similarity'] == pytest.approx(1.0) for row in plus)


def test_cosine_tie_break_is_lexical_and_manifest_order_independent():
    base = _base_rows()[:1]
    manifest = list(reversed(_manifest_rows()))
    embeddings = _embeddings()
    embeddings['a/train/good/ref-b.jpg'] = np.asarray([1.0, 0.0], dtype=np.float32)

    first = retrieve_top1_references(base, manifest, embeddings)
    second = retrieve_top1_references(base, list(reversed(manifest)), embeddings)

    assert first == second
    assert first[0]['retrieved_reference_image'] == 'a/train/good/ref-a.jpg'


def test_provenance_validation_rejects_leaky_reference():
    base = _base_rows()
    manifest = _manifest_rows() + [{
        'relative_path': 'a/train/good/leaky.jpg',
        'asset_category': 'a',
        'split': 'train',
        'label': 'normal',
        'source_label_raw': 'good',
        'content_fingerprint': '10:q1',
    }]
    matches = retrieve_top1_references(base, manifest, _embeddings())
    plus = build_one_shot_plus_rows(base, matches)
    plus[0]['reference_image'] = 'a/train/good/leaky.jpg'

    with pytest.raises(ValueError, match='not leakage-safe train-normal'):
        validate_retrieval_provenance(base, plus, matches, manifest)


def test_provenance_validation_rejects_non_finite_similarity():
    base = _base_rows()
    manifest = _manifest_rows()
    matches = retrieve_top1_references(base, manifest, _embeddings())
    plus = build_one_shot_plus_rows(base, matches)
    plus[0]['reference_similarity'] = float('nan')

    with pytest.raises(ValueError, match='similarity is not finite'):
        validate_retrieval_provenance(base, plus, matches, manifest)


class _FakeEncoder:
    def __init__(self, provenance: EmbeddingProvenance) -> None:
        self.provenance = provenance
        self.calls: list[list[str]] = []

    def encode(self, images):
        paths = [str(image) for image in images]
        self.calls.append(paths)
        return np.asarray(
            [[float(len(path)), float(sum(path.encode('utf-8')) % 17 + 1)] for path in paths],
            dtype=np.float32,
        )


def _cache_rows():
    return [
        {
            'relative_path': f'a/train/good/{name}.jpg',
            'asset_category': 'a',
            'content_fingerprint': f'{index}:{name}',
        }
        for index, name in enumerate(('one', 'two', 'three'), 1)
    ]


def test_npz_cache_is_pickle_free_and_encodes_only_incremental_misses(tmp_path: Path):
    cache_path = tmp_path / 'embeddings.npz'
    provenance = EmbeddingProvenance(model_id='fake/model', model_revision='rev-1')
    rows = _cache_rows()
    first_encoder = _FakeEncoder(provenance)

    first = embed_rows_with_cache(
        rows[:2],
        encoder=first_encoder,
        image_loader=lambda row: row['relative_path'],
        cache_path=cache_path,
        batch_size=1,
    )

    assert set(first) == {row['relative_path'] for row in rows[:2]}
    assert first_encoder.calls == [
        ['a/train/good/one.jpg'],
        ['a/train/good/two.jpg'],
    ]
    with np.load(cache_path, allow_pickle=False) as cache:
        assert all(cache[name].dtype.kind != 'O' for name in cache.files)
        assert cache['paths'].shape == (2,)
    assert not cache_path.with_name(cache_path.name + '.tmp').exists()

    second_encoder = _FakeEncoder(provenance)
    second = embed_rows_with_cache(
        rows,
        encoder=second_encoder,
        image_loader=lambda row: row['relative_path'],
        cache_path=cache_path,
        batch_size=2,
    )

    assert set(second) == {row['relative_path'] for row in rows}
    assert second_encoder.calls == [['a/train/good/three.jpg']]
    with np.load(cache_path, allow_pickle=False) as cache:
        assert cache['paths'].shape == (3,)


def test_npz_cache_checkpoints_at_bounded_batch_intervals(tmp_path: Path, monkeypatch):
    cache_path = tmp_path / 'embeddings.npz'
    rows = [
        {
            'relative_path': f'a/train/good/{index}.jpg',
            'asset_category': 'a',
            'content_fingerprint': f'{index}:fp',
        }
        for index in range(6)
    ]
    encoder = _FakeEncoder(EmbeddingProvenance(model_id='fake/model'))
    writes = []
    original_write = NpzEmbeddingCache._write

    def counted_write(cache):
        writes.append(len(cache._records))
        original_write(cache)

    monkeypatch.setattr(NpzEmbeddingCache, '_write', counted_write)
    embed_rows_with_cache(
        rows,
        encoder=encoder,
        image_loader=lambda row: row['relative_path'],
        cache_path=cache_path,
        batch_size=1,
        checkpoint_every_batches=2,
    )

    assert writes == [2, 4, 6]
    assert not cache_path.with_name(cache_path.name + '.tmp').exists()


def test_npz_cache_rejects_model_provenance_mismatch(tmp_path: Path):
    cache_path = tmp_path / 'embeddings.npz'
    rows = _cache_rows()[:1]
    first_encoder = _FakeEncoder(EmbeddingProvenance(model_id='fake/a'))
    embed_rows_with_cache(
        rows,
        encoder=first_encoder,
        image_loader=lambda row: row['relative_path'],
        cache_path=cache_path,
    )

    mismatched = NpzEmbeddingCache(
        cache_path,
        EmbeddingProvenance(model_id='fake/b'),
    )
    with pytest.raises(ValueError, match='cache provenance mismatch'):
        mismatched.get(rows[0])

    revision_mismatch = NpzEmbeddingCache(
        cache_path,
        EmbeddingProvenance(model_id='fake/a', model_revision='different-revision'),
    )
    with pytest.raises(ValueError, match='cache provenance mismatch'):
        revision_mismatch.get(rows[0])


def test_atomic_cache_replace_retries_transient_windows_lock(tmp_path: Path, monkeypatch):
    source = tmp_path / 'cache.npz.tmp'
    target = tmp_path / 'cache.npz'
    source.write_bytes(b'new')
    target.write_bytes(b'old')
    real_replace = retrieval.os.replace
    attempts = []

    def flaky_replace(from_path, to_path):
        attempts.append((from_path, to_path))
        if len(attempts) < 3:
            raise PermissionError('transient file lock')
        real_replace(from_path, to_path)

    monkeypatch.setattr(retrieval.os, 'replace', flaky_replace)
    monkeypatch.setattr(retrieval.time, 'sleep', lambda _seconds: None)

    retrieval._atomic_replace(source, target)

    assert len(attempts) == 3
    assert target.read_bytes() == b'new'
    assert not source.exists()


def test_dinov2_encoder_is_lazy_and_does_not_download_on_construction():
    encoder = DinoV2Encoder(device='cpu')

    assert encoder.provenance.model_id == 'facebook/dinov2-base'
    assert encoder.provenance.model_revision == DEFAULT_MODEL_REVISION
    assert encoder._model is None
    assert encoder._processor is None
