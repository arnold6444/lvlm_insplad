#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_gold_baselines as base  # noqa: E402
import run_gold_v2_baselines as v2  # noqa: E402
from insplad_benchmark.hf_adapters import infer_hf, load_hf_adapter  # noqa: E402
from insplad_benchmark.vision_expert import read_manifest, resolve_artifact_path  # noqa: E402

EXPERTS = ("patchcore", "dinomaly", "anomalyclip", "reconpatch_lite")
CONDITIONS = ("score", "roi", "heatmap", "score_roi", "score_heatmap")
MODELS = tuple(base.MODEL_REGISTRY)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    base.write_jsonl(path, [dict(row) for row in rows])


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    base.write_csv(path, [dict(row) for row in rows])


def upsert_rows(path: Path, new_rows: Sequence[Mapping[str, object]], keys: Sequence[str]) -> None:
    old_rows = read_csv(path)
    replace = {tuple(str(row.get(k, "")) for k in keys) for row in new_rows}
    kept = [
        row for row in old_rows
        if tuple(str(row.get(k, "")) for k in keys) not in replace
    ]
    write_csv(path, [*kept, *new_rows])


def resolve_score(row: Mapping[str, str]) -> float:
    for key in ("model_pred_score", "anomaly_score_top1pct"):
        value = str(row.get(key, "")).strip()
        if value:
            try:
                return float(value)
            except ValueError:
                pass
    raise ValueError(
        f"missing numeric score for {row.get('expert')}/{row.get('query_id')}"
    )


def add_score_percentiles(rows: list[dict[str, str]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["expert"]), str(row["asset_category"]))].append(row)
    for _, group in grouped.items():
        values = pd.Series([resolve_score(row) for row in group], dtype=float)
        ranks = values.rank(method="average", pct=True) * 100.0
        for row, percentile in zip(group, ranks.tolist(), strict=True):
            row["score_percentile_full"] = f"{percentile:.6f}"


def prompt(row: Mapping[str, object], condition: str, expert: str, percentile: float) -> str:
    options = "\n".join(f"{k}. {v}" for k, v in dict(row["options"]).items())
    score_text = (
        f"The {expert} vision-expert anomaly score is at the "
        f"{percentile:.1f}th percentile among same-asset images in this fixed "
        "unlabeled benchmark cohort. Higher is more suspicious. "
        "This relative score uses no ground-truth label and may be wrong. "
    )
    if condition == "score":
        evidence = score_text
        image_desc = "Image 1 is the complete TARGET image."
    elif condition == "roi":
        evidence = (
            f"Image 2 is a candidate ROI crop selected automatically by {expert}. "
            "It may highlight a normal region. "
        )
        image_desc = "Image 1 is the complete TARGET image."
    elif condition == "heatmap":
        evidence = (
            f"Image 2 is the {expert} anomaly heatmap overlay of the TARGET; "
            "warmer regions are more suspicious and the yellow box marks the "
            "automatically selected candidate ROI. It may be noisy. "
        )
        image_desc = "Image 1 is the complete TARGET image."
    elif condition == "score_roi":
        evidence = score_text + (
            f"Image 2 is a candidate ROI crop selected automatically by {expert}. "
            "Use both cues only as hints. "
        )
        image_desc = "Image 1 is the complete TARGET image."
    elif condition == "score_heatmap":
        evidence = score_text + (
            f"Image 2 is the {expert} anomaly heatmap overlay of the TARGET. "
            "Use both cues only as hints. "
        )
        image_desc = "Image 1 is the complete TARGET image."
    else:
        raise ValueError(condition)

    return (
        "You are an industrial visual inspector. "
        f"{image_desc} {evidence}"
        "Answer the multiple-choice question about the TARGET. "
        "Do not assume the vision expert is correct.\n\n"
        f"Question: {row['question']}\n{options}\n\n"
        "Return exactly one option letter and nothing else."
    )


def evidence_images(
    condition: str,
    target: Path,
    expert_row: Mapping[str, str],
    manifest_path: Path,
    artifact_root: Path,
) -> list[Path]:
    if condition == "score":
        return [target]
    if condition in {"roi", "score_roi"}:
        evidence = resolve_artifact_path(
            manifest_path, str(expert_row["roi_path"]), artifact_root
        )
        return [target, evidence]
    if condition in {"heatmap", "score_heatmap"}:
        evidence = resolve_artifact_path(
            manifest_path, str(expert_row["overlay_path"]), artifact_root
        )
        return [target, evidence]
    raise ValueError(condition)


def infer_luxia(images: list[Path], prompt_text: str, api_key: str) -> tuple[str, float]:
    from insplad_benchmark.luxia_adapter import infer_luxia_pipeline

    started = time.time()
    result = infer_luxia_pipeline(images, prompt_text, api_key)
    return str(result["output"]), round(time.time() - started, 3)


def prediction_reusable(
    row: Mapping[str, object] | None,
    *,
    model_key: str,
    expert: str,
    condition: str,
) -> bool:
    if not row or row.get("error") or not str(row.get("output", "")).strip():
        return False
    return (
        str(row.get("model_key", "")) == model_key
        and str(row.get("expert", "")) == expert
        and str(row.get("condition", "")) == condition
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Final gold-v2 Vision Expert evidence ablation across all LVLMs."
    )
    parser.add_argument("--qa-dir", type=Path, required=True)
    parser.add_argument("--sup-zip", type=Path, required=True)
    parser.add_argument("--roi-manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--experts", nargs="+", default=list(EXPERTS), choices=list(EXPERTS))
    parser.add_argument(
        "--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS)
    )
    parser.add_argument("--checkpoint-every", type=int, default=5)
    args = parser.parse_args()

    rows = v2.load_gold_qa_v2(args.qa_dir)
    core_ids = {str(row["core_id"]) for row in rows}
    if len(core_ids) != 246:
        raise RuntimeError(f"expected 246 unique gold-v2 images, got {len(core_ids)}")

    manifest_rows = [dict(row) for row in read_manifest(args.roi_manifest)]
    selected = [
        row
        for row in manifest_rows
        if str(row.get("expert", "")) in set(args.experts)
        and str(row.get("query_id", "")) in core_ids
    ]
    expected_manifest = len(core_ids) * len(args.experts)
    if len(selected) != expected_manifest:
        raise RuntimeError(
            f"ROI manifest coverage mismatch: {len(selected)} != {expected_manifest}"
        )
    add_score_percentiles(selected)
    expert_index = {
        (str(row["query_id"]), str(row["expert"])): row for row in selected
    }
    if len(expert_index) != expected_manifest:
        raise RuntimeError("ROI manifest contains duplicate expert/query keys")

    for expert_row in selected:
        for field in ("roi_path", "overlay_path"):
            path = resolve_artifact_path(
                args.roi_manifest, str(expert_row[field]), args.artifact_root
            )
            if not path.is_file():
                raise FileNotFoundError(path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    target_paths = base.prepare_images(rows, args.sup_zip, args.out_dir / "_images" / "target")
    leaderboard_path = args.out_dir / "leaderboard.csv"
    by_asset_path = args.out_dir / "metrics_by_asset.csv"
    status_path = args.out_dir / "run_status.json"

    status_json = []
    if status_path.is_file():
        try:
            status_json = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            status_json = []

    q3_counts = Counter(
        str(row["asset_category"])
        for row in rows
        if row["question_type"] == "defect_localization"
    )
    luxia_key = os.environ.get("LUXIA_API_KEY", "")

    for model_key in args.models:
        spec = base.MODEL_REGISTRY[model_key]
        adapter = None
        if spec["backend"] == "hf":
            print(f"=== load {model_key} ===", flush=True)
            adapter = load_hf_adapter(spec)
        elif model_key == "luxia" and not luxia_key:
            raise RuntimeError("LUXIA_API_KEY is required for Luxia")

        try:
            for expert in args.experts:
                for condition in args.conditions:
                    print(f"\n=== {model_key} / {expert} / {condition} ===", flush=True)
                    pred_path = (
                        args.out_dir
                        / "predictions"
                        / model_key
                        / f"{expert}__{condition}.jsonl"
                    )
                    existing_rows = read_jsonl(pred_path)
                    existing = {str(row["id"]): row for row in existing_rows}
                    predictions: list[dict] = []
                    called = 0
                    errors = 0

                    for index, row in enumerate(rows, start=1):
                        old = existing.get(str(row["id"]))
                        if prediction_reusable(
                            old,
                            model_key=model_key,
                            expert=expert,
                            condition=condition,
                        ):
                            predictions.append(old)
                            continue

                        erow = expert_index[(str(row["core_id"]), expert)]
                        target = target_paths[str(row["image"])]
                        images = evidence_images(
                            condition, target, erow, args.roi_manifest, args.artifact_root
                        )
                        percentile = float(erow["score_percentile_full"])
                        prompt_text = prompt(row, condition, expert, percentile)
                        started = time.time()
                        output = ""
                        error = ""
                        latency = 0.0
                        try:
                            if spec["backend"] == "hf":
                                output = str(infer_hf(adapter, images, prompt_text))
                                latency = round(time.time() - started, 3)
                            else:
                                output, latency = infer_luxia(images, prompt_text, luxia_key)
                        except Exception as exc:
                            error = (
                                f"{type(exc).__name__}: {exc}\n"
                                f"{traceback.format_exc(limit=3)}"
                            )
                            latency = round(time.time() - started, 3)
                            errors += 1

                        choice = base.parse_choice(
                            output, {str(k).upper() for k in row["options"]}
                        )
                        result = {
                            "id": row["id"],
                            "core_id": row["core_id"],
                            "asset_category": row["asset_category"],
                            "question_type": row["question_type"],
                            "model_key": model_key,
                            "model_id": spec["model_id"],
                            "expert": expert,
                            "condition": condition,
                            "score_percentile_full": percentile,
                            "output": output,
                            "predicted_choice": choice,
                            "latency_s": latency,
                            "error": error,
                        }
                        predictions.append(result)
                        called += 1
                        if called % args.checkpoint_every == 0 or index == len(rows):
                            write_jsonl(pred_path, predictions)

                    write_jsonl(pred_path, predictions)
                    metric, assets = base.evaluate_model(model_key, rows, pred_path)
                    metric.update({"expert": expert, "condition": condition})
                    for asset_metric in assets:
                        asset_metric.update(
                            {
                                "expert": expert,
                                "condition": condition,
                                "q3_questions": q3_counts.get(
                                    asset_metric["asset_category"], 0
                                ),
                            }
                        )
                        if asset_metric["q3_questions"] == 0:
                            asset_metric["q3_accuracy"] = None

                    upsert_rows(
                        leaderboard_path,
                        [metric],
                        ("model_key", "expert", "condition"),
                    )
                    upsert_rows(
                        by_asset_path,
                        assets,
                        ("model_key", "expert", "condition", "asset_category"),
                    )
                    status = {
                        "model_key": model_key,
                        "expert": expert,
                        "condition": condition,
                        "status": "complete",
                        "called_this_run": called,
                        "row_errors": errors,
                    }
                    replace_key = (model_key, expert, condition)
                    status_json = [
                        item for item in status_json
                        if (
                            str(item.get("model_key", "")),
                            str(item.get("expert", "")),
                            str(item.get("condition", "")),
                        ) != replace_key
                    ]
                    status_json.append(status)
                    status_path.write_text(
                        json.dumps(status_json, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
        finally:
            adapter = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    leaderboard = read_csv(leaderboard_path)
    selected_lb = [
        row for row in leaderboard
        if row.get("model_key") in set(args.models)
        and row.get("expert") in set(args.experts)
        and row.get("condition") in set(args.conditions)
    ]
    expected_runs = len(args.models) * len(args.experts) * len(args.conditions)
    if len(selected_lb) != expected_runs:
        raise RuntimeError(
            f"leaderboard coverage mismatch: {len(selected_lb)} != {expected_runs}"
        )

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in selected_lb:
        grouped[(row["expert"], row["condition"])].append(row)
    summary = []
    for (expert, condition), group in grouped.items():
        n = len(group)
        anomaly = sum(float(row["anomaly_f1"]) for row in group) / n
        defect = sum(float(row["defect_macro_f1"]) for row in group) / n
        summary.append(
            {
                "expert": expert,
                "condition": condition,
                "models": n,
                "mean_overall_accuracy": sum(
                    float(row["overall_accuracy"]) for row in group
                ) / n,
                "mean_anomaly_f1": anomaly,
                "mean_defect_macro_f1": defect,
                "mean_localization_accuracy": sum(
                    float(row["localization_accuracy"]) for row in group
                ) / n,
                "mean_primary_score": (anomaly + defect) / 2.0,
            }
        )
    summary.sort(
        key=lambda row: (row["mean_primary_score"], row["mean_overall_accuracy"]),
        reverse=True,
    )
    write_csv(args.out_dir / "screening_summary.csv", summary)
    metadata = {
        "protocol": "gold_v2_vision_expert_full",
        "gold_images": 246,
        "qa_questions": len(rows),
        "models": args.models,
        "experts": args.experts,
        "conditions": args.conditions,
        "score_definition": (
            "within-expert, within-asset percentile over the fixed unlabeled "
            "246-image benchmark cohort; no GT labels used (transductive relative cue)"
        ),
        "selection_rule": (
            "report all combinations; summarize by mean of anomaly F1 and defect macro-F1 "
            "across all five LVLMs; localization reported separately"
        ),
        "expected_runs": expected_runs,
    }
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary[:10], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
