from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_gold_baselines.py"
SPEC = importlib.util.spec_from_file_location("gold_baselines", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _master_rows() -> list[dict]:
    rows = []
    for i in range(250):
        base = {
            "core_id": f"core-{i}",
            "asset_category": "vari-grip",
            "image": f"defect_supervised/vari-grip/val/good/{i}.jpg",
            "anomaly_gt": "normal" if i % 2 == 0 else "anomalous",
            "defect_gt": "normal" if i % 2 == 0 else "rust",
        }
        rows.append({
            **base,
            "id": f"core-{i}-q1",
            "question_type": "anomaly_detection",
            "question": "Normal or anomalous?",
            "options": {"A": "Normal.", "B": "Anomalous."},
            "answer": "A" if i % 2 == 0 else "B",
            "answer_text": "Normal." if i % 2 == 0 else "Anomalous.",
        })
        rows.append({
            **base,
            "id": f"core-{i}-q2",
            "question_type": "defect_classification",
            "question": "Defect?",
            "options": {
                "A": "Normal / no defect.",
                "B": "Rust / corrosion.",
                "C": "Missing cap / missing insulator component.",
                "D": "Bird nest / foreign nesting material.",
            },
            "answer": "A" if i % 2 == 0 else "B",
            "answer_text": "Normal / no defect." if i % 2 == 0 else "Rust / corrosion.",
        })
    return rows


def test_parse_choice_is_strict_but_tolerates_common_wrappers():
    assert RUNNER.parse_choice("A", {"A", "B"}) == "A"
    assert RUNNER.parse_choice("Answer: B", {"A", "B"}) == "B"
    assert RUNNER.parse_choice('{"answer":"C"}', {"A", "B", "C"}) == "C"
    assert RUNNER.parse_choice("A or B", {"A", "B"}) is None
    assert RUNNER.parse_choice("", {"A", "B"}) is None


def test_load_gold_qa_merges_master_and_q3_only(tmp_path):
    master = _master_rows()
    _write_jsonl(tmp_path / "master_qa_500.jsonl", master)

    loc_rows = [
        master[0],
        master[1],
        {
            **master[0],
            "id": "core-0-q3",
            "question_type": "defect_localization",
            "question": "Where?",
            "options": {
                "A": "Upper-left region.",
                "B": "Center region.",
                "C": "No localized defect region is present.",
                "D": "Lower-right region.",
            },
            "answer": "C",
            "answer_text": "No localized defect region is present.",
        },
    ]
    _write_jsonl(tmp_path / "localization_qa_3.jsonl", loc_rows)

    rows = RUNNER.load_gold_qa(tmp_path)
    assert len(rows) == 501
    assert sum(r["question_type"] == "anomaly_detection" for r in rows) == 250
    assert sum(r["question_type"] == "defect_classification" for r in rows) == 250
    assert sum(r["question_type"] == "defect_localization" for r in rows) == 1


def test_evaluation_metrics_use_semantic_labels(tmp_path):
    rows = [
        {
            "id": "q1a",
            "core_id": "a",
            "asset_category": "vari-grip",
            "question_type": "anomaly_detection",
            "anomaly_gt": "normal",
            "defect_gt": "normal",
            "options": {"A": "Normal.", "B": "Anomalous."},
            "answer": "A",
        },
        {
            "id": "q1b",
            "core_id": "b",
            "asset_category": "vari-grip",
            "question_type": "anomaly_detection",
            "anomaly_gt": "anomalous",
            "defect_gt": "rust",
            "options": {"A": "Normal.", "B": "Anomalous."},
            "answer": "B",
        },
        {
            "id": "q2a",
            "core_id": "a",
            "asset_category": "vari-grip",
            "question_type": "defect_classification",
            "anomaly_gt": "normal",
            "defect_gt": "normal",
            "options": {
                "A": "Normal / no defect.",
                "B": "Rust / corrosion.",
                "C": "Missing cap / missing insulator component.",
                "D": "Bird nest / foreign nesting material.",
            },
            "answer": "A",
        },
        {
            "id": "q2b",
            "core_id": "b",
            "asset_category": "vari-grip",
            "question_type": "defect_classification",
            "anomaly_gt": "anomalous",
            "defect_gt": "rust",
            "options": {
                "A": "Normal / no defect.",
                "B": "Rust / corrosion.",
                "C": "Missing cap / missing insulator component.",
                "D": "Bird nest / foreign nesting material.",
            },
            "answer": "B",
        },
        {
            "id": "q3b",
            "core_id": "b",
            "asset_category": "vari-grip",
            "question_type": "defect_localization",
            "anomaly_gt": "anomalous",
            "defect_gt": "rust",
            "options": {
                "A": "Upper-left region.",
                "B": "Center region.",
                "C": "No localized defect region is present.",
                "D": "Lower-right region.",
            },
            "answer": "B",
        },
    ]
    preds = [
        {"id": row["id"], "predicted_choice": row["answer"], "output": row["answer"], "error": None}
        for row in rows
    ]
    pred_path = tmp_path / "pred.jsonl"
    _write_jsonl(pred_path, preds)

    metric, by_asset = RUNNER.evaluate_model("qwen3_5_4b", rows, pred_path)
    assert metric["overall_accuracy"] == 1.0
    assert metric["anomaly_f1"] == 1.0
    assert metric["defect_macro_f1"] == 1.0
    assert metric["localization_accuracy"] == 1.0
    assert by_asset[0]["q3_accuracy"] == 1.0
