import csv
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_aligned_core_results.py"
spec = importlib.util.spec_from_file_location("result_analysis", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_anomaly_summary_counts_and_metrics():
    gt = [
        {"id": "n1", "answer": "normal"},
        {"id": "n2", "answer": "normal"},
        {"id": "a1", "answer": "anomalous"},
        {"id": "a2", "answer": "anomalous"},
    ]
    outputs = {
        "n1": '{"anomaly":"normal"}',
        "n2": '{"anomaly":"anomalous"}',
        "a1": '{"anomaly":"anomalous"}',
        "a2": '{"anomaly":"normal"}',
    }
    row = module.anomaly_summary(gt, outputs, "zero_shot")
    assert (row["tp"], row["fp"], row["tn"], row["fn"]) == (1, 1, 1, 1)
    assert row["precision"] == 0.5
    assert row["recall"] == 0.5
    assert row["specificity"] == 0.5
    assert row["balanced_accuracy"] == 0.5
    assert row["f1"] == 0.5


def test_structured_summary_exposes_imbalance_baseline():
    gt = [
        {"id": "n1", "valid_defect_classes": ["normal", "rust"], "answer": {"anomaly": "normal", "defect_class": "normal"}},
        {"id": "n2", "valid_defect_classes": ["normal", "rust"], "answer": {"anomaly": "normal", "defect_class": "normal"}},
        {"id": "a1", "valid_defect_classes": ["normal", "rust"], "answer": {"anomaly": "anomalous", "defect_class": "rust"}},
    ]
    outputs = {
        "n1": '{"anomaly":"normal","defect_class":"normal"}',
        "n2": '{"anomaly":"normal","defect_class":"normal"}',
        "a1": '{"anomaly":"normal","defect_class":"normal"}',
    }
    row, per_defect = module.structured_summary(gt, outputs)
    assert row["joint_accuracy"] == 2 / 3
    assert row["normal_only_joint_baseline"] == 2 / 3
    assert row["joint_gain_over_normal_only"] == 0.0
    assert {item["defect_class"] for item in per_defect} == {"normal", "rust"}


def test_anomaly_qa_alignment_fails_when_ground_truth_changes():
    zero = [{"id": "q1", "image": "test/q1.jpg", "asset_category": "asset", "answer": "normal"}]
    retrieved = [{"id": "q1", "image": "test/q1.jpg", "asset_category": "asset", "answer": "anomalous"}]

    try:
        module.validate_anomaly_qa_alignment(zero, retrieved, "one_shot_plus")
    except ValueError as exc:
        assert "field answer" in str(exc)
    else:
        raise AssertionError("misaligned QA must fail")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_analyzer_joins_separate_plus_results_and_reports_yoke_fp(tmp_path, monkeypatch):
    qa_dir = tmp_path / "qa"
    baseline_dir = tmp_path / "baseline"
    plus_dir = tmp_path / "one_shot_plus_results"
    out_dir = tmp_path / "diagnostics"
    model_dir = baseline_dir / "qwen3_5_4b"

    anomaly_rows = [
        {"id": "y-n", "image": "test/y-n.jpg", "asset_category": "yoke-suspension", "answer": "normal"},
        {"id": "y-a", "image": "test/y-a.jpg", "asset_category": "yoke-suspension", "answer": "anomalous"},
        {"id": "g-n", "image": "test/g-n.jpg", "asset_category": "glass-insulator", "answer": "normal"},
        {"id": "g-a", "image": "test/g-a.jpg", "asset_category": "glass-insulator", "answer": "anomalous"},
    ]
    fixed_rows = [{**row, "reference_image": f"train/{row['asset_category']}.jpg"} for row in anomaly_rows]
    retrieved_rows = [
        {
            **row,
            "reference_image": f"train/retrieved-{row['asset_category']}.jpg",
            "retrieval_method": "dinov2_cosine_top1",
        }
        for row in anomaly_rows
    ]
    structured_rows = [
        {
            "id": row["id"],
            "image": row["image"],
            "asset_category": row["asset_category"],
            "valid_defect_classes": ["normal", "rust"],
            "answer": {
                "anomaly": row["answer"],
                "defect_class": "normal" if row["answer"] == "normal" else "rust",
            },
        }
        for row in anomaly_rows
    ]
    _write_jsonl(qa_dir / "zero_shot.jsonl", anomaly_rows)
    _write_jsonl(qa_dir / "one_shot.jsonl", fixed_rows)
    _write_jsonl(qa_dir / "structured.jsonl", structured_rows)
    plus_qa = qa_dir / "one_shot_plus.jsonl"
    _write_jsonl(plus_qa, retrieved_rows)

    metadata = {"model_key": "qwen3_5_4b", "model_id": "Qwen/Qwen3.5-4B"}
    zero_outputs = {
        "y-n": "normal", "y-a": "normal", "g-n": "normal", "g-a": "anomalous",
    }
    fixed_outputs = {
        "y-n": "anomalous", "y-a": "anomalous", "g-n": "normal", "g-a": "anomalous",
    }
    retrieved_outputs = {
        "y-n": "normal", "y-a": "anomalous", "g-n": "normal", "g-a": "anomalous",
    }
    structured_outputs = {
        row["id"]: json.dumps(row["answer"])
        for row in structured_rows
    }

    def prediction_rows(outputs, *, plus=False):
        rows = [{"id": row_id, "output": output, **metadata} for row_id, output in outputs.items()]
        if plus:
            gt_by_id = {row["id"]: row for row in retrieved_rows}
            for row in rows:
                row.update({
                    "reference_image": gt_by_id[row["id"]]["reference_image"],
                    "retrieval_method": gt_by_id[row["id"]]["retrieval_method"],
                })
        return rows

    _write_jsonl(model_dir / "zero_shot.jsonl", prediction_rows(zero_outputs))
    _write_jsonl(model_dir / "one_shot.jsonl", prediction_rows(fixed_outputs))
    _write_jsonl(model_dir / "structured.jsonl", prediction_rows(structured_outputs))
    _write_jsonl(
        plus_dir / "qwen3_5_4b" / "one_shot_plus.jsonl",
        prediction_rows(retrieved_outputs, plus=True),
    )

    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "analyze_aligned_core_results.py",
            "--qa-dir", str(qa_dir),
            "--results-dir", str(baseline_dir),
            "--one-shot-plus-qa", str(plus_qa),
            "--one-shot-plus-results-dir", str(plus_dir),
            "--out-dir", str(out_dir),
        ],
    )
    assert module.main() == 0

    overall = _read_csv(out_dir / "anomaly_overall.csv")
    assert {row["regime"] for row in overall} == {"zero_shot", "one_shot", "one_shot_plus"}
    macro = _read_csv(out_dir / "anomaly_asset_macro.csv")
    assert {row["regime"] for row in macro} == {"zero_shot", "one_shot", "one_shot_plus"}

    yoke = _read_csv(out_dir / "yoke_fp_comparison.csv")
    assert len(yoke) == 1
    assert (int(yoke[0]["fixed_fp"]), int(yoke[0]["retrieved_fp"]), int(yoke[0]["delta_fp"])) == (1, 0, -1)

    by_asset = _read_csv(out_dir / "fixed_vs_retrieved_by_asset.csv")
    assert {row["asset_category"] for row in by_asset} == {"glass-insulator", "yoke-suspension"}


def test_one_shot_plus_predictions_reject_partial_results(tmp_path):
    plus_gt = [
        {
            "id": "q1",
            "reference_image": "asset/train/ref-1.jpg",
            "retrieval_method": "dinov2_cosine_top1",
        },
        {
            "id": "q2",
            "reference_image": "asset/train/ref-2.jpg",
            "retrieval_method": "dinov2_cosine_top1",
        },
    ]
    path = tmp_path / "one_shot_plus.jsonl"
    _write_jsonl(path, [{
        "id": "q1",
        "output": "normal",
        "model_key": "qwen3_5_4b",
        "model_id": "Qwen/Qwen3.5-4B",
        "reference_image": "asset/train/ref-1.jpg",
        "retrieval_method": "dinov2_cosine_top1",
    }])

    with pytest.raises(ValueError, match="must cover exactly all 2 one_shot_plus QA IDs once"):
        module.validated_one_shot_plus_outputs(
            path,
            plus_gt,
            expected_model_key="qwen3_5_4b",
            expected_model_id="Qwen/Qwen3.5-4B",
        )


@pytest.mark.parametrize(
    ("metadata_override", "mismatched_field"),
    [
        ({"model_key": "internvl3_5_8b"}, "model_key"),
        ({"model_id": "OpenGVLab/InternVL3_5-8B"}, "model_id"),
    ],
)
def test_one_shot_plus_predictions_reject_wrong_model_identity(
    tmp_path, metadata_override, mismatched_field
):
    plus_gt = [{
        "id": "q1",
        "reference_image": "asset/train/ref-1.jpg",
        "retrieval_method": "dinov2_cosine_top1",
    }]
    path = tmp_path / "one_shot_plus.jsonl"
    prediction = {
        "id": "q1",
        "output": "normal",
        "model_key": "qwen3_5_4b",
        "model_id": "Qwen/Qwen3.5-4B",
        "reference_image": "asset/train/ref-1.jpg",
        "retrieval_method": "dinov2_cosine_top1",
    }
    prediction.update(metadata_override)
    _write_jsonl(path, [prediction])

    with pytest.raises(ValueError, match=f"expected baseline {mismatched_field}="):
        module.validated_one_shot_plus_outputs(
            path,
            plus_gt,
            expected_model_key="qwen3_5_4b",
            expected_model_id="Qwen/Qwen3.5-4B",
        )


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("reference_image", "asset/train/wrong.jpg"),
        ("retrieval_method", "wrong_method"),
    ],
)
def test_one_shot_plus_predictions_reject_wrong_retrieval_provenance(
    tmp_path, field, wrong_value
):
    plus_gt = [{
        "id": "q1",
        "reference_image": "asset/train/ref-1.jpg",
        "retrieval_method": "dinov2_cosine_top1",
    }]
    prediction = {
        "id": "q1",
        "output": "normal",
        "model_key": "qwen3_5_4b",
        "model_id": "Qwen/Qwen3.5-4B",
        "reference_image": "asset/train/ref-1.jpg",
        "retrieval_method": "dinov2_cosine_top1",
    }
    prediction[field] = wrong_value
    path = tmp_path / "one_shot_plus.jsonl"
    _write_jsonl(path, [prediction])

    with pytest.raises(ValueError, match=f"expected QA {field}="):
        module.validated_one_shot_plus_outputs(
            path,
            plus_gt,
            expected_model_key="qwen3_5_4b",
            expected_model_id="Qwen/Qwen3.5-4B",
        )


def test_one_shot_plus_rejects_unsupported_baseline_model():
    path = Path("llava_onevision_7b/one_shot_plus.jsonl")

    with pytest.raises(ValueError, match="unsupported one_shot_plus model_key 'llava_onevision_7b'"):
        module.validate_one_shot_plus_model_key("llava_onevision_7b", path)
