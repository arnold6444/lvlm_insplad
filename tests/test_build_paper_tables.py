from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_paper_tables_t", ROOT / "scripts" / "build_paper_tables.py")
TABLES = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(TABLES)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_build_tables_marks_best_and_handles_lower_is_better(tmp_path):
    cons = tmp_path / "results" / "gold_v2_consistency_metrics"
    rows = []
    for cond, q1, contra in (("image_only", 0.6, 0.30), ("normal_reference", 0.8, 0.20),
                             ("vx:patchcore:score", 0.7, 0.35), ("vx:dinomaly:roi", 0.5, 0.40),
                             ("ref+patchcore:score", 0.75, 0.25)):
        rows.append({"model_key": "qwen3_5_4b", "condition": cond, "q1_anomaly_f1": q1, "q2_defect_macro_f1": 0.5,
                     "q3_accuracy": 0.6, "consistency_error": contra, "q1q3_contradiction": 0.3,
                     "joint_accuracy": 0.45, "false_defect_rate": 0.4})
    _write(cons / "consistency_leaderboard.csv", rows)
    (cons / "run_metadata.json").write_text(json.dumps({"q3_constant_baselines": {
        "questions": 42, "always_no_defect_region": 0.524, "always_center": 0.476,
        "answer_regions": {"Center region.": 20}}}), encoding="utf-8")
    _write(cons / "significance_main.csv", [
        {"model_key": "qwen3_5_4b", "baseline": "image_only", "condition": "normal_reference",
         "metric": "q1_anomaly_f1", "delta": 0.2, "ci_low": 0.1, "ci_high": 0.3, "significant": "True"}])
    out = tmp_path / "tables"
    info = TABLES.build(tmp_path / "results", out)
    assert info["rows"] == 5 and info["significance_rows"] == 1
    md = (out / "README.md").read_text(encoding="utf-8")
    assert "| Qwen3.5-4B | + Normal ref | **0.800** |" in md          # best Q1 bolded
    assert "**0.200**" in md                                            # lowest contradiction bolded
    assert "+0.200 [+0.100, +0.300]*" in md
    with (out / "table3_vision_expert.csv").open(encoding="utf-8") as h:
        t3 = list(csv.DictReader(h))
    assert [r["Expert"] for r in t3] == ["patchcore", "dinomaly"]       # sorted by Q1
    for name in ("table1_main.csv", "table2_reference.csv", "table4_q3_baselines.csv", "all_conditions.csv"):
        assert (out / name).is_file()
