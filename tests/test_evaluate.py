import csv
import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
spec = importlib.util.spec_from_file_location("evaluate_script", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_one_shot_plus_cli_writes_confusion_and_unweighted_asset_macro(tmp_path, monkeypatch):
    qa = [
        {"id": "n1", "asset_category": "asset-a", "answer": "normal"},
        {"id": "a1", "asset_category": "asset-a", "answer": "anomalous"},
        {"id": "n2", "asset_category": "asset-b", "answer": "normal"},
        {"id": "n3", "asset_category": "asset-b", "answer": "normal"},
        {"id": "a2", "asset_category": "asset-b", "answer": "anomalous"},
    ]
    predictions = [
        {"id": "n1", "output": '{"anomaly":"normal"}'},
        {"id": "a1", "output": '{"anomaly":"anomalous"}'},
        {"id": "n2", "output": '{"anomaly":"anomalous"}'},
        {"id": "n3", "output": '{"anomaly":"normal"}'},
        {"id": "a2", "output": '{"anomaly":"normal"}'},
    ]
    qa_path = tmp_path / "one_shot_plus.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    out_dir = tmp_path / "metrics"
    write_jsonl(qa_path, qa)
    write_jsonl(prediction_path, predictions)

    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "evaluate.py",
            "--task", "anomaly",
            "--qa", str(qa_path),
            "--predictions", str(prediction_path),
            "--out-dir", str(out_dir),
            "--regime", "one_shot_plus",
        ],
    )
    assert module.main() == 0

    overall = read_csv(out_dir / "anomaly_overall.csv")[0]
    assert overall["regime"] == "one_shot_plus"
    assert tuple(int(overall[key]) for key in ("tp", "fp", "tn", "fn")) == (1, 1, 2, 1)
    assert float(overall["precision"]) == 0.5
    assert float(overall["recall"]) == 0.5
    assert float(overall["specificity"]) == 2 / 3

    macro = read_csv(out_dir / "anomaly_asset_macro.csv")[0]
    assert int(macro["asset_n"]) == 2
    assert float(macro["balanced_accuracy"]) == 0.625
    assert float(macro["f1"]) == 0.5


def test_upsert_csv_unions_old_and_additive_schema(tmp_path):
    path = tmp_path / "metrics.csv"
    module.write_csv(path, [{"regime": "zero_shot", "balanced_accuracy": 0.5}])
    module.upsert_csv(
        path,
        [{"regime": "one_shot_plus", "balanced_accuracy": 0.75, "precision": 0.8}],
        ("regime",),
    )
    rows = read_csv(path)
    assert [row["regime"] for row in rows] == ["zero_shot", "one_shot_plus"]
    assert "precision" in rows[0]
    assert rows[0]["precision"] == ""
    assert rows[1]["precision"] == "0.8"
