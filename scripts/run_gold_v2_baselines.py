from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "scripts" / "run_gold_baselines.py"
SPEC = importlib.util.spec_from_file_location("run_gold_baselines", LEGACY)
base = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(base)


def load_gold_qa_v2(qa_dir: Path) -> list[dict]:
    master_candidates = sorted(qa_dir.glob("master_qa_*.jsonl"))
    if not master_candidates:
        raise FileNotFoundError(f"master_qa_*.jsonl not found under {qa_dir}")
    master = base.read_jsonl(master_candidates[-1])
    types = Counter(str(r.get("question_type")) for r in master)
    q1 = types.get("anomaly_detection", 0)
    q2 = types.get("defect_classification", 0)
    if q1 <= 0 or q1 != q2 or len(master) != q1 + q2:
        raise RuntimeError(f"unexpected master question counts: {dict(types)}")

    q3: list[dict] = []
    loc_candidates = sorted(qa_dir.glob("localization_qa_*.jsonl"))
    if loc_candidates:
        all_loc = base.read_jsonl(loc_candidates[-1])
        q3 = [r for r in all_loc if r.get("question_type") == "defect_localization"]

    rows = [dict(r) for r in master] + [dict(r) for r in q3]
    rows.sort(
        key=lambda r: (
            str(r.get("core_id", "")),
            base.QUESTION_ORDER.get(str(r.get("question_type", "")), 99),
            str(r.get("id", "")),
        )
    )
    ids = [str(r.get("id", "")) for r in rows]
    if any(not x for x in ids) or len(ids) != len(set(ids)):
        raise RuntimeError("QA IDs must be present and unique")
    return rows


base.load_gold_qa = load_gold_qa_v2


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--qa-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    known, _ = parser.parse_known_args()

    rc = base.main()
    if rc != 0:
        return rc

    rows = load_gold_qa_v2(known.qa_dir)
    metadata_path = known.out_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    counts = Counter(str(r.get("question_type")) for r in rows)
    metadata.update(
        {
            "master_q1_q2_count": counts["anomaly_detection"] + counts["defect_classification"],
            "q1_count": counts["anomaly_detection"],
            "q2_count": counts["defect_classification"],
            "q3_count": counts["defect_localization"],
            "condition": "image_only_final_gold_v2",
            "benchmark_version": "gold_v2_final",
        }
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
