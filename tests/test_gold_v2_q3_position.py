import csv
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import score_gold_v2_q3_position as pos


def loc_row(core, boxes, defect='rust'):
    return dict(core_id=core, protocol='gold_v2_loc_mmad', defect_gt=defect, asset_category='asset',
                localization_gt=dict(normalized_bboxes_xywh=boxes, basis='manual_box_annotation'))


def answer(policy, core, text, kind='defect_localization'):
    return dict(policy=policy, core_id=core, question_type=kind, predicted_text=text)


def write_inputs(tmp_path, texts):
    loc = [loc_row('a', [[.05, .05, .1, .1], [.85, .85, .1, .1]]), loc_row('b', [[.4, .4, .1, .1]], 'missing-cap')]
    (tmp_path / 'loc.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in loc), encoding='utf-8')
    rows = [answer(p, core, texts[p][core]) for p in pos.POLICIES for core in ('a', 'b', 'n')]
    rows.append(answer('baseline', 'a', 'Normal.', 'anomaly_detection'))
    with open(tmp_path / 'answers.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_hit_counts_box_center_cells(tmp_path):
    none = 'No localized defect region is present.'
    texts = {p: {'a': 'Lower-right region.', 'b': 'Center region.', 'n': none} for p in pos.POLICIES}
    texts['baseline'] = {'a': 'Upper-center region.', 'b': none, 'n': 'Center region.'}
    write_inputs(tmp_path, texts)
    assert pos.main(['--answers', str(tmp_path / 'answers.csv'), '--loc-qa', str(tmp_path / 'loc.jsonl'),
                     '--out-dir', str(tmp_path / 'out')]) == 0
    board = {r['policy']: r for r in csv.DictReader(open(tmp_path / 'out' / 'leaderboard.csv', encoding='utf-8'))}
    # 'a' has boxes centred in cells 1 and 9, 'b' in cell 5; the normal image 'n' has no location row.
    assert (board['previous_agent']['position_hit'], board['previous_agent']['exact_cell_set']) == ('2', '1')
    assert (board['baseline']['region_reported'], board['baseline']['position_hit']) == ('1', '0')
    assert board['always_cell_5']['position_hit'] == '1' and board['always_cell_9']['position_hit'] == '1'
    meta = json.loads((tmp_path / 'out' / 'run_metadata.json').read_text(encoding='utf-8'))
    assert meta['new_lvlm_inference'] is False and meta['location_images'] == 2


def test_rejects_unknown_answer_text(tmp_path):
    texts = {p: {'a': 'Somewhere.', 'b': 'Center region.', 'n': 'Center region.'} for p in pos.POLICIES}
    write_inputs(tmp_path, texts)
    with pytest.raises(ValueError, match='unparsed'):
        pos.main(['--answers', str(tmp_path / 'answers.csv'), '--loc-qa', str(tmp_path / 'loc.jsonl'),
                  '--out-dir', str(tmp_path / 'out')])
