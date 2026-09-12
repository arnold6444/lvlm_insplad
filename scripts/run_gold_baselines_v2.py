from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_gold_baselines as base


def load_gold_qa(qa_dir: Path) -> list[dict]:
    master_candidates = sorted(Path(qa_dir).glob('master_qa_*.jsonl'))
    if not master_candidates:
        raise FileNotFoundError(f'master_qa_*.jsonl not found under {qa_dir}')
    master = base.read_jsonl(master_candidates[-1])
    types = Counter(str(row.get('question_type')) for row in master)
    q1 = types.get('anomaly_detection', 0)
    q2 = types.get('defect_classification', 0)
    unexpected = set(types) - {'anomaly_detection', 'defect_classification'}
    if q1 <= 0 or q1 != q2 or len(master) != q1 + q2 or unexpected:
        raise RuntimeError(f'invalid Q1/Q2 master benchmark: rows={len(master)}, counts={dict(types)}')

    q3: list[dict] = []
    loc_candidates = sorted(Path(qa_dir).glob('localization_qa_*.jsonl'))
    if loc_candidates:
        all_loc = base.read_jsonl(loc_candidates[-1])
        q3 = [row for row in all_loc if row.get('question_type') == 'defect_localization']
        master_core_ids = {str(row.get('core_id')) for row in master}
        unknown = sorted({str(row.get('core_id')) for row in q3} - master_core_ids)
        if unknown:
            raise RuntimeError(f'Q3 contains core IDs outside master benchmark: {unknown[:10]}')

    rows = [dict(row) for row in master] + [dict(row) for row in q3]
    rows.sort(key=lambda row: (
        str(row.get('core_id', '')),
        base.QUESTION_ORDER.get(str(row.get('question_type', '')), 99),
        str(row.get('id', '')),
    ))
    ids = [str(row.get('id', '')) for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise RuntimeError('QA IDs must be present and unique')
    return rows


base.load_gold_qa = load_gold_qa

if __name__ == '__main__':
    raise SystemExit(base.main())
