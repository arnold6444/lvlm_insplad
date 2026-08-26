#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from insplad_benchmark.metrics import (
    balanced_accuracy,
    binary_f1,
    consistency_error,
    invalid_output_rate,
    macro_f1,
    per_class_prf,
)
from insplad_benchmark.parsing import parse_anomaly_output, parse_structured_output

INVALID = "<invalid>"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_div(num: int | float, den: int | float) -> float:
    return 0.0 if den == 0 else num / den


def anomaly_summary(gt_rows: list[dict], outputs: dict[str, str], regime: str) -> dict:
    parsed = [parse_anomaly_output(outputs.get(row["id"], "")) for row in gt_rows]
    y_true = [str(row["answer"]) for row in gt_rows]
    y_pred = [item.anomaly if item.valid and item.anomaly else INVALID for item in parsed]

    tp = sum(t == "anomalous" and p == "anomalous" for t, p in zip(y_true, y_pred))
    fp = sum(t == "normal" and p == "anomalous" for t, p in zip(y_true, y_pred))
    tn = sum(t == "normal" and p == "normal" for t, p in zip(y_true, y_pred))
    fn = sum(t == "anomalous" and p != "anomalous" for t, p in zip(y_true, y_pred))
    normal_n = sum(t == "normal" for t in y_true)
    anomalous_n = sum(t == "anomalous" for t in y_true)
    invalid_normal_n = sum(t == "normal" and p == INVALID for t, p in zip(y_true, y_pred))
    invalid_anomalous_n = sum(t == "anomalous" and p == INVALID for t, p in zip(y_true, y_pred))

    return {
        "regime": regime,
        "n": len(gt_rows),
        "normal_n": normal_n,
        "anomalous_n": anomalous_n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, anomalous_n),
        "specificity": safe_div(tn, normal_n),
        "predicted_anomalous_n": sum(p == "anomalous" for p in y_pred),
        "predicted_normal_n": sum(p == "normal" for p in y_pred),
        "invalid_normal_n": invalid_normal_n,
        "invalid_anomalous_n": invalid_anomalous_n,
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, ["normal", "anomalous"]) if y_true else 0.0,
        "f1": binary_f1(y_true, y_pred) if y_true else 0.0,
        "invalid_output_rate": invalid_output_rate([item.valid for item in parsed]),
    }


def structured_summary(gt_rows: list[dict], outputs: dict[str, str]) -> tuple[dict, list[dict]]:
    parsed = [
        parse_structured_output(outputs.get(row["id"], ""), set(row["valid_defect_classes"]))
        for row in gt_rows
    ]
    anomaly_true = [row["answer"]["anomaly"] for row in gt_rows]
    defect_true = [row["answer"]["defect_class"] for row in gt_rows]
    anomaly_pred = [item.anomaly if item.valid else INVALID for item in parsed]
    defect_pred = [item.defect_class if item.valid else INVALID for item in parsed]
    labels = sorted(set(defect_true))

    joint_correct_n = sum(
        a_t == a_p and d_t == d_p
        for a_t, a_p, d_t, d_p in zip(anomaly_true, anomaly_pred, defect_true, defect_pred)
    )
    normal_indices = [idx for idx, value in enumerate(anomaly_true) if value == "normal"]
    false_defect_n = sum(
        parsed[idx].valid and defect_pred[idx] not in {"normal", INVALID}
        for idx in normal_indices
    )
    valid_pairs = [(a, d) for a, d, item in zip(anomaly_pred, defect_pred, parsed) if item.valid]
    consistency_error_n = sum(
        (a == "normal" and d != "normal") or (a == "anomalous" and d == "normal")
        for a, d in valid_pairs
    )

    overall = {
        "n": len(gt_rows),
        "normal_n": len(normal_indices),
        "anomalous_n": len(gt_rows) - len(normal_indices),
        "valid_n": sum(item.valid for item in parsed),
        "joint_correct_n": joint_correct_n,
        "joint_accuracy": safe_div(joint_correct_n, len(gt_rows)),
        "normal_only_joint_baseline": safe_div(len(normal_indices), len(gt_rows)),
        "joint_gain_over_normal_only": safe_div(joint_correct_n, len(gt_rows)) - safe_div(len(normal_indices), len(gt_rows)),
        "defect_balanced_accuracy": balanced_accuracy(defect_true, defect_pred, labels) if defect_true else 0.0,
        "defect_macro_f1": macro_f1(defect_true, defect_pred, labels) if defect_true else 0.0,
        "false_defect_n": false_defect_n,
        "false_defect_rate": safe_div(false_defect_n, len(normal_indices)),
        "consistency_error_n": consistency_error_n,
        "consistency_error": consistency_error([a for a, _ in valid_pairs], [d for _, d in valid_pairs]) if valid_pairs else 0.0,
        "invalid_output_rate": invalid_output_rate([item.valid for item in parsed]),
    }
    per_defect = [
        {"defect_class": label, **stats}
        for label, stats in per_class_prf(defect_true, defect_pred, labels).items()
    ]
    return overall, per_defect


def outputs_by_id(path: Path) -> tuple[dict[str, str], dict]:
    rows = read_jsonl(path)
    outputs = {row["id"]: str(row.get("output", "")) for row in rows}
    meta = rows[0] if rows else {}
    return outputs, meta


def grouped(rows: list[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        result[str(row["asset_category"])].append(row)
    return result


def complete_model_dirs(results_dir: Path) -> list[Path]:
    required = ("zero_shot.jsonl", "one_shot.jsonl", "structured.jsonl")
    return sorted(
        path for path in results_dir.iterdir()
        if path.is_dir() and all((path / name).is_file() for name in required)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze completed Aligned Core LVLM predictions without rerunning inference.")
    parser.add_argument("--qa-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    out_dir = args.out_dir or (args.results_dir / "diagnostics")
    zero_gt = read_jsonl(args.qa_dir / "zero_shot.jsonl")
    one_gt = read_jsonl(args.qa_dir / "one_shot.jsonl")
    structured_gt = read_jsonl(args.qa_dir / "structured.jsonl")

    anomaly_overall: list[dict] = []
    anomaly_by_asset: list[dict] = []
    structured_overall: list[dict] = []
    structured_by_asset: list[dict] = []
    defect_by_class: list[dict] = []
    reference_shift: list[dict] = []

    for model_dir in complete_model_dirs(args.results_dir):
        zero_outputs, zero_meta = outputs_by_id(model_dir / "zero_shot.jsonl")
        one_outputs, one_meta = outputs_by_id(model_dir / "one_shot.jsonl")
        structured_outputs, structured_meta = outputs_by_id(model_dir / "structured.jsonl")
        meta = zero_meta or one_meta or structured_meta
        model_key = str(meta.get("model_key") or model_dir.name)
        model_id = str(meta.get("model_id") or "")

        zero = anomaly_summary(zero_gt, zero_outputs, "zero_shot")
        one = anomaly_summary(one_gt, one_outputs, "one_shot")
        for row in (zero, one):
            anomaly_overall.append({"model_key": model_key, "model_id": model_id, **row})

        zero_groups = grouped(zero_gt)
        one_groups = grouped(one_gt)
        for asset in sorted(zero_groups):
            anomaly_by_asset.append({
                "model_key": model_key, "model_id": model_id, "asset_category": asset,
                **anomaly_summary(zero_groups[asset], zero_outputs, "zero_shot"),
            })
            anomaly_by_asset.append({
                "model_key": model_key, "model_id": model_id, "asset_category": asset,
                **anomaly_summary(one_groups[asset], one_outputs, "one_shot"),
            })

        structured, per_defect = structured_summary(structured_gt, structured_outputs)
        structured_overall.append({"model_key": model_key, "model_id": model_id, **structured})
        for row in per_defect:
            defect_by_class.append({"model_key": model_key, "model_id": model_id, **row})

        for asset, rows in sorted(grouped(structured_gt).items()):
            asset_summary, _ = structured_summary(rows, structured_outputs)
            structured_by_asset.append({
                "model_key": model_key, "model_id": model_id, "asset_category": asset, **asset_summary,
            })

        reference_shift.append({
            "model_key": model_key,
            "model_id": model_id,
            "delta_bacc": one["balanced_accuracy"] - zero["balanced_accuracy"],
            "delta_f1": one["f1"] - zero["f1"],
            "delta_precision": one["precision"] - zero["precision"],
            "delta_recall": one["recall"] - zero["recall"],
            "delta_specificity": one["specificity"] - zero["specificity"],
            "delta_predicted_anomalous_n": one["predicted_anomalous_n"] - zero["predicted_anomalous_n"],
        })

    write_csv(out_dir / "anomaly_overall.csv", anomaly_overall)
    write_csv(out_dir / "anomaly_by_asset.csv", anomaly_by_asset)
    write_csv(out_dir / "structured_overall.csv", structured_overall)
    write_csv(out_dir / "structured_by_asset.csv", structured_by_asset)
    write_csv(out_dir / "defect_by_class.csv", defect_by_class)
    write_csv(out_dir / "reference_shift.csv", reference_shift)

    print(f"Analyzed {len(complete_model_dirs(args.results_dir))} completed model directories -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
