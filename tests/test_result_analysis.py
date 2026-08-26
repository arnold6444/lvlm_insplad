import importlib.util
from pathlib import Path

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
