#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from insplad_benchmark.metrics import (
    anomaly_classification_summary,
    balanced_accuracy,
    consistency_error,
    invalid_output_rate,
    macro_f1,
    per_class_prf,
)
from insplad_benchmark.parsing import parse_anomaly_output, parse_structured_output

INVALID = "<invalid>"
ONE_SHOT_PLUS_MODEL_KEYS = frozenset({
    "qwen3_5_4b",
    "internvl3_5_8b",
    "minicpm_v_4_5",
})


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
    return {"regime": regime, **anomaly_classification_summary(y_true, y_pred)}


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


def validated_one_shot_plus_outputs(
    path: Path,
    plus_gt: list[dict],
    *,
    expected_model_key: str,
    expected_model_id: str,
) -> dict[str, str]:
    rows = read_jsonl(path)
    expected_by_id = {str(row["id"]): row for row in plus_gt}
    prediction_ids = [str(row.get("id", "")) for row in rows]

    duplicate_counts = Counter(prediction_ids)
    if len(duplicate_counts) != len(prediction_ids):
        duplicates = sorted(row_id for row_id, count in duplicate_counts.items() if count > 1)
        raise ValueError(f"{path} contains duplicate one_shot_plus prediction IDs: {duplicates[:10]}")

    missing = sorted(set(expected_by_id) - set(prediction_ids))
    extra = sorted(set(prediction_ids) - set(expected_by_id))
    if len(rows) != len(plus_gt) or missing or extra:
        raise ValueError(
            f"{path} must cover exactly all {len(plus_gt)} one_shot_plus QA IDs once; "
            f"got {len(rows)}, missing={missing[:10]}, extra={extra[:10]}"
        )

    for index, row in enumerate(rows):
        row_id = prediction_ids[index]
        for field, expected_value in (
            ("model_key", expected_model_key),
            ("model_id", expected_model_id),
        ):
            actual_value = str(row.get(field, ""))
            if actual_value != expected_value:
                raise ValueError(
                    f"{path} row {row_id} has {field}={actual_value!r}; "
                    f"expected baseline {field}={expected_value!r}"
                )

        expected = expected_by_id[row_id]
        for field in ("reference_image", "retrieval_method"):
            if field not in row:
                raise ValueError(f"{path} row {row_id} is missing required {field}")
            if row[field] != expected.get(field):
                raise ValueError(
                    f"{path} row {row_id} has {field}={row[field]!r}; "
                    f"expected QA {field}={expected.get(field)!r}"
                )

    return {row_id: str(row.get("output", "")) for row_id, row in zip(prediction_ids, rows)}


def validate_one_shot_plus_model_key(model_key: str, path: Path) -> None:
    if model_key not in ONE_SHOT_PLUS_MODEL_KEYS:
        raise ValueError(
            f"unsupported one_shot_plus model_key {model_key!r} in {path}; "
            "allowed model keys are qwen3_5_4b, internvl3_5_8b, minicpm_v_4_5"
        )


def grouped(rows: list[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        result[str(row["asset_category"])].append(row)
    return result


def validate_anomaly_qa_alignment(reference: list[dict], candidate: list[dict], candidate_name: str) -> None:
    fields = ("id", "image", "asset_category", "answer")
    if len(reference) != len(candidate):
        raise ValueError(
            f"zero_shot and {candidate_name} QA counts differ: {len(reference)} != {len(candidate)}"
        )

    reference_ids = [str(row.get("id", "")) for row in reference]
    candidate_ids = [str(row.get("id", "")) for row in candidate]
    if len(set(reference_ids)) != len(reference_ids):
        raise ValueError("zero_shot QA contains duplicate IDs")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError(f"{candidate_name} QA contains duplicate IDs")

    for index, (expected, actual) in enumerate(zip(reference, candidate)):
        for field in fields:
            if expected.get(field) != actual.get(field):
                raise ValueError(
                    f"zero_shot and {candidate_name} QA differ at row {index}, field {field}: "
                    f"{expected.get(field)!r} != {actual.get(field)!r}"
                )


def asset_macro_summary(rows: list[dict], regime: str) -> dict:
    return {
        "regime": regime,
        "asset_n": len(rows),
        "balanced_accuracy": safe_div(
            sum(float(row["balanced_accuracy"]) for row in rows), len(rows)
        ),
        "f1": safe_div(sum(float(row["f1"]) for row in rows), len(rows)),
    }


COMPARISON_METRICS = (
    "balanced_accuracy",
    "f1",
    "precision",
    "recall",
    "specificity",
    "tp",
    "fp",
    "tn",
    "fn",
    "invalid_output_rate",
    "invalid_normal_n",
    "invalid_anomalous_n",
)


def fixed_vs_retrieved_summary(fixed: dict, retrieved: dict) -> dict:
    row: dict[str, float | int] = {}
    for metric in COMPARISON_METRICS:
        fixed_value = fixed[metric]
        retrieved_value = retrieved[metric]
        row[f"fixed_{metric}"] = fixed_value
        row[f"retrieved_{metric}"] = retrieved_value
        row[f"delta_{metric}"] = retrieved_value - fixed_value
    return row


def complete_model_dirs(results_dir: Path) -> list[Path]:
    required = ("zero_shot.jsonl", "one_shot.jsonl", "structured.jsonl")
    return sorted(
        path for path in results_dir.iterdir()
        if path.is_dir() and all((path / name).is_file() for name in required)
    )


def one_shot_plus_prediction_path(results_dir: Path, model_dir: Path, model_key: str) -> Path | None:
    candidates = [
        results_dir / model_dir.name / "one_shot_plus.jsonl",
        results_dir / model_key / "one_shot_plus.jsonl",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze completed Aligned Core LVLM predictions without rerunning inference.")
    parser.add_argument("--qa-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--one-shot-plus-qa", type=Path)
    parser.add_argument("--one-shot-plus-results-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    if bool(args.one_shot_plus_qa) != bool(args.one_shot_plus_results_dir):
        parser.error("--one-shot-plus-qa and --one-shot-plus-results-dir must be provided together")

    out_dir = args.out_dir or (args.results_dir / "diagnostics")
    zero_gt = read_jsonl(args.qa_dir / "zero_shot.jsonl")
    one_gt = read_jsonl(args.qa_dir / "one_shot.jsonl")
    structured_gt = read_jsonl(args.qa_dir / "structured.jsonl")
    validate_anomaly_qa_alignment(zero_gt, one_gt, "one_shot")
    plus_gt = read_jsonl(args.one_shot_plus_qa) if args.one_shot_plus_qa else None
    if plus_gt is not None:
        validate_anomaly_qa_alignment(zero_gt, plus_gt, "one_shot_plus")

    zero_groups = grouped(zero_gt)
    one_groups = grouped(one_gt)
    plus_groups = grouped(plus_gt) if plus_gt is not None else None
    model_dirs = complete_model_dirs(args.results_dir)

    anomaly_overall: list[dict] = []
    anomaly_by_asset: list[dict] = []
    anomaly_asset_macro: list[dict] = []
    structured_overall: list[dict] = []
    structured_by_asset: list[dict] = []
    defect_by_class: list[dict] = []
    reference_shift: list[dict] = []
    fixed_vs_retrieved_overall: list[dict] = []
    fixed_vs_retrieved_by_asset: list[dict] = []
    matched_plus_models = 0

    for model_dir in model_dirs:
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

        zero_asset_rows: list[dict] = []
        one_asset_rows: list[dict] = []
        for asset in sorted(zero_groups):
            zero_asset = anomaly_summary(zero_groups[asset], zero_outputs, "zero_shot")
            one_asset = anomaly_summary(one_groups[asset], one_outputs, "one_shot")
            zero_asset_rows.append(zero_asset)
            one_asset_rows.append(one_asset)
            anomaly_by_asset.extend([
                {"model_key": model_key, "model_id": model_id, "asset_category": asset, **zero_asset},
                {"model_key": model_key, "model_id": model_id, "asset_category": asset, **one_asset},
            ])

        anomaly_asset_macro.extend([
            {"model_key": model_key, "model_id": model_id, **asset_macro_summary(zero_asset_rows, "zero_shot")},
            {"model_key": model_key, "model_id": model_id, **asset_macro_summary(one_asset_rows, "one_shot")},
        ])

        if args.one_shot_plus_results_dir is not None and plus_gt is not None and plus_groups is not None:
            plus_path = one_shot_plus_prediction_path(
                args.one_shot_plus_results_dir, model_dir, model_key
            )
            if plus_path is not None:
                validate_one_shot_plus_model_key(model_key, plus_path)
                plus_outputs = validated_one_shot_plus_outputs(
                    plus_path,
                    plus_gt,
                    expected_model_key=model_key,
                    expected_model_id=model_id,
                )
                retrieved = anomaly_summary(plus_gt, plus_outputs, "one_shot_plus")
                anomaly_overall.append({"model_key": model_key, "model_id": model_id, **retrieved})
                fixed_vs_retrieved_overall.append({
                    "model_key": model_key,
                    "model_id": model_id,
                    **fixed_vs_retrieved_summary(one, retrieved),
                })

                retrieved_asset_rows: list[dict] = []
                for asset, fixed_asset in zip(sorted(zero_groups), one_asset_rows):
                    retrieved_asset = anomaly_summary(
                        plus_groups[asset], plus_outputs, "one_shot_plus"
                    )
                    retrieved_asset_rows.append(retrieved_asset)
                    anomaly_by_asset.append({
                        "model_key": model_key,
                        "model_id": model_id,
                        "asset_category": asset,
                        **retrieved_asset,
                    })
                    fixed_vs_retrieved_by_asset.append({
                        "model_key": model_key,
                        "model_id": model_id,
                        "asset_category": asset,
                        **fixed_vs_retrieved_summary(fixed_asset, retrieved_asset),
                    })
                anomaly_asset_macro.append({
                    "model_key": model_key,
                    "model_id": model_id,
                    **asset_macro_summary(retrieved_asset_rows, "one_shot_plus"),
                })
                matched_plus_models += 1

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

    if args.one_shot_plus_results_dir is not None and matched_plus_models == 0:
        raise RuntimeError(
            "no one_shot_plus.jsonl files matched completed baseline model directories under "
            f"{args.one_shot_plus_results_dir}"
        )

    yoke_fp_comparison = [
        {
            "model_key": row["model_key"],
            "model_id": row["model_id"],
            "asset_category": row["asset_category"],
            "fixed_fp": row["fixed_fp"],
            "retrieved_fp": row["retrieved_fp"],
            "delta_fp": row["delta_fp"],
            "fixed_specificity": row["fixed_specificity"],
            "retrieved_specificity": row["retrieved_specificity"],
            "delta_specificity": row["delta_specificity"],
            "fixed_invalid_normal_n": row["fixed_invalid_normal_n"],
            "retrieved_invalid_normal_n": row["retrieved_invalid_normal_n"],
        }
        for row in fixed_vs_retrieved_by_asset
        if row["asset_category"] == "yoke-suspension"
    ]

    write_csv(out_dir / "anomaly_overall.csv", anomaly_overall)
    write_csv(out_dir / "anomaly_by_asset.csv", anomaly_by_asset)
    write_csv(out_dir / "anomaly_asset_macro.csv", anomaly_asset_macro)
    write_csv(out_dir / "structured_overall.csv", structured_overall)
    write_csv(out_dir / "structured_by_asset.csv", structured_by_asset)
    write_csv(out_dir / "defect_by_class.csv", defect_by_class)
    write_csv(out_dir / "reference_shift.csv", reference_shift)
    write_csv(out_dir / "fixed_vs_retrieved_overall.csv", fixed_vs_retrieved_overall)
    write_csv(out_dir / "fixed_vs_retrieved_by_asset.csv", fixed_vs_retrieved_by_asset)
    write_csv(out_dir / "yoke_fp_comparison.csv", yoke_fp_comparison)

    suffix = f", {matched_plus_models} with one_shot_plus" if plus_gt is not None else ""
    print(f"Analyzed {len(model_dirs)} completed model directories{suffix} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
