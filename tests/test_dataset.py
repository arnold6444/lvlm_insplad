import csv
import json
from pathlib import Path

from insplad_benchmark.dataset import (
    aligned_core_rows,
    build_anomaly_qa,
    build_defect_qa,
    category_intersection,
    core_link_rows,
    load_overlap_confirmation,
    normalize_defect,
    parse_annotation_previews,
    parse_supervised_manifest,
    parse_unsupervised_manifest,
    select_reference_by_asset,
)


def write_manifest(path: Path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['path', 'is_dir', 'size', 'compressed_size', 'extension', 'crc32'])
        w.writeheader()
        w.writerows(rows)


def test_normalization_and_structure(tmp_path):
    u = tmp_path / 'u.csv'
    s = tmp_path / 's.csv'
    write_manifest(u, [
        {'path': 'asset-a/train/good/ref.jpg', 'is_dir': False, 'size': 10, 'compressed_size': 9, 'extension': '.jpg', 'crc32': 'aaaa'},
        {'path': 'asset-a/test/good/a.jpg', 'is_dir': False, 'size': 11, 'compressed_size': 9, 'extension': '.jpg', 'crc32': 'bbbb'},
        {'path': 'asset-a/test/rust/b.jpg', 'is_dir': False, 'size': 12, 'compressed_size': 9, 'extension': '.jpg', 'crc32': 'cccc'},
        {'path': 'asset-b/test/good/c.jpg', 'is_dir': False, 'size': 13, 'compressed_size': 9, 'extension': '.jpg', 'crc32': 'dddd'},
    ])
    write_manifest(s, [
        {'path': 'defect_supervised/asset-a/train/normal/n.jpg', 'is_dir': False, 'size': 20, 'compressed_size': 9, 'extension': '.jpg', 'crc32': 'eeee'},
        {'path': 'defect_supervised/asset-a/train/corrosão/x.jpg', 'is_dir': False, 'size': 21, 'compressed_size': 9, 'extension': '.jpg', 'crc32': 'ffff'},
        {'path': 'defect_supervised/asset-a/val/good/v.jpg', 'is_dir': False, 'size': 22, 'compressed_size': 9, 'extension': '.jpg', 'crc32': '1111'},
    ])
    ur = parse_unsupervised_manifest(u)
    sr, _ = parse_supervised_manifest(s)
    assert category_intersection(ur, sr) == ['asset-a']
    assert normalize_defect('corrosão') == 'corrosao'
    assert {r['defect_class'] for r in sr} == {'normal', 'corrosao'}
    qa, split, reason = build_defect_qa(sr)
    assert split == 'val' and 'no test' in reason
    assert qa[0]['answer'] == {'anomaly': 'normal', 'defect_class': 'normal'}


def test_reference_excludes_test_duplicate(tmp_path):
    u = tmp_path / 'u.csv'
    write_manifest(u, [
        {'path': 'asset-a/train/good/leaky.jpg', 'is_dir': False, 'size': 10, 'compressed_size': 9, 'extension': '.jpg', 'crc32': 'aaaa'},
        {'path': 'asset-a/train/good/safe.jpg', 'is_dir': False, 'size': 20, 'compressed_size': 9, 'extension': '.jpg', 'crc32': 'bbbb'},
        {'path': 'asset-a/test/good/test.jpg', 'is_dir': False, 'size': 10, 'compressed_size': 9, 'extension': '.jpg', 'crc32': 'aaaa'},
    ])
    rows = parse_unsupervised_manifest(u)
    refs = select_reference_by_asset(rows, ['asset-a'], seed=42)
    assert refs['asset-a']['relative_path'].endswith('safe.jpg')
    z = build_anomaly_qa(rows, ['asset-a'])
    o = build_anomaly_qa(rows, ['asset-a'], references=refs)
    assert [(r['id'], r['image'], r['answer']) for r in z] == [(r['id'], r['image'], r['answer']) for r in o]


def test_overlap_summary_must_confirm_all_candidates(tmp_path):
    path = tmp_path / 'summary.json'
    path.write_text(json.dumps({
        'candidate_count': 2,
        'sha256_match_count': 2,
        'sha256_mismatch_count': 0,
    }), encoding='utf-8')
    summary = load_overlap_confirmation(path, 2)
    assert summary['all_candidates_sha256_confirmed'] is True


def test_aligned_core_excludes_binary_gt_conflict():
    overlap = [
        {
            'unsupervised_split': 'test', 'supervised_split': 'val',
            'unsupervised_anomaly': 'normal', 'supervised_anomaly': 'normal',
            'asset_category': 'a', 'unsupervised_path': 'a/test/good/x.jpg',
            'supervised_path': 'defect_supervised/a/val/good/x.jpg',
            'unsupervised_label_raw': 'good', 'supervised_defect_raw': 'good',
            'supervised_defect': 'normal', 'filename': 'x.jpg',
        },
        {
            'unsupervised_split': 'test', 'supervised_split': 'val',
            'unsupervised_anomaly': 'normal', 'supervised_anomaly': 'anomalous',
            'asset_category': 'a', 'unsupervised_path': 'a/test/good/y.jpg',
            'supervised_path': 'defect_supervised/a/val/rust/y.jpg',
            'unsupervised_label_raw': 'good', 'supervised_defect_raw': 'rust',
            'supervised_defect': 'rust', 'filename': 'y.jpg',
        },
    ]
    clean, conflicts = aligned_core_rows(overlap, exact_sha256_confirmed=True)
    assert len(clean) == 1
    assert len(conflicts) == 1
    assert clean[0]['core_anomaly'] == 'normal'


def test_core_link_joins_track_a_and_b_ids():
    clean = [{
        'asset_category': 'a',
        'unsupervised_path': 'a/test/good/x.jpg',
        'supervised_path': 'defect_supervised/a/val/good/x.jpg',
        'core_anomaly': 'normal',
        'supervised_defect': 'normal',
        'unsupervised_label_raw': 'good',
        'supervised_defect_raw': 'good',
    }]
    refs = {'a': {'relative_path': 'a/train/good/ref.jpg'}}
    links = core_link_rows(clean, refs)
    assert links[0]['anomaly_id'].startswith('anomaly-')
    assert links[0]['structured_id'].startswith('defect-')
    assert links[0]['reference_image'] == 'a/train/good/ref.jpg'


def test_annotation_preview_class_ids_are_not_binary_anomaly(tmp_path):
    path = tmp_path / 'annotations.json'
    path.write_text(json.dumps([{
        'path': 'defect_supervised/vari-grip/annotations/image_list_val.txt',
        'truncated': False,
        'preview': 'good/a.jpg 1\nbird-nest/b.jpg 0\nrust/c.jpg 2\n',
    }]), encoding='utf-8')
    info = parse_annotation_previews(path)
    assert info['line_schema'] == '<relative_image_path> <integer_class_id>'
    assert info['numeric_class_id_is_binary_anomaly'] is False
    assert info['by_asset']['vari-grip']['raw_defect_to_class_ids'] == {
        'bird-nest': [0], 'good': [1], 'rust': [2]
    }
