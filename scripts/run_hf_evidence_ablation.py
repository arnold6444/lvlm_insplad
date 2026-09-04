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
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from insplad_benchmark.evidence_ablation import EVIDENCE_CONDITIONS, read_jsonl, write_csv_atomic
from insplad_benchmark.hf_adapters import infer_hf, load_hf_adapter
from insplad_benchmark.metrics import anomaly_classification_summary
from insplad_benchmark.parsing import parse_anomaly_output

MODEL_REGISTRY = {
    "qwen3_5_4b": {
        "backend": "hf", "provider": "huggingface", "model_id": "Qwen/Qwen3.5-4B",
        "trust_remote_code": False, "adapter": "pipeline", "role": "qwen_family",
    },
    "internvl3_5_8b": {
        "backend": "hf", "provider": "huggingface", "model_id": "OpenGVLab/InternVL3_5-8B",
        "trust_remote_code": True, "adapter": "internvl_chat", "max_total_tiles": 6,
        "role": "internvl_family",
    },
}
INVALID = "<invalid>"


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()
        try: os.fsync(handle.fileno())
        except OSError: pass
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def workspace_queries(root: Path) -> dict[str, Path]:
    result = {}
    for row in read_csv(root / "workspace_manifest.csv"):
        if row.get("role") == "query":
            result[row["query_id"]] = root / row["local_path"]
    return result


def safe_extract_member(zip_path: Path, member: str, cache: Path) -> Path:
    normalized = member.replace("\\", "/").lstrip("/")
    out = cache / normalized
    if out.is_file(): return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as archive:
        names = {name.replace("\\", "/").lstrip("/"): name for name in archive.namelist()}
        if normalized not in names: raise FileNotFoundError(member)
        out.write_bytes(archive.read(names[normalized]))
    return out


def prompt(condition: str, expert: str, score_percentile: float) -> str:
    common = (
        "Classify only the TARGET industrial asset as normal or anomalous. "
        "Vision-expert evidence is only a hint and can be wrong. "
        "Your first character must be '{'. Return exactly one JSON object: "
        '{"anomaly":"normal"} or {"anomaly":"anomalous"}. No explanation.'
    )
    score = (
        f" The {expert} expert score is at the {score_percentile:.1f}th percentile "
        "within this fixed unlabeled screening batch; higher means more suspicious. "
        "This percentile is a screening-only relative cue, not a ground-truth label."
    )
    if condition == "score":
        return "Image 1 is the TARGET." + score + " " + common
    if condition == "bbox":
        return (
            f"Image 1 is the TARGET with a yellow box marking the most suspicious region selected by {expert}. "
            "The box may still contain a normal region. " + common
        )
    if condition == "heatmap":
        return (
            f"Image 1 is the TARGET with a {expert} anomaly heatmap blended on top; warmer regions are more suspicious. "
            "The heatmap may be noisy. " + common
        )
    if condition == "reference_score":
        return "Image 1 is a NORMAL same-category reference. Image 2 is the TARGET." + score + " Compare them. " + common
    if condition == "reference_bbox":
        return (
            f"Image 1 is a NORMAL same-category reference. Image 2 is the TARGET with a yellow {expert} box. "
            "Compare them and use the box only as an attention hint. " + common
        )
    if condition == "reference_heatmap":
        return (
            f"Image 1 is a NORMAL same-category reference. Image 2 is the TARGET with a {expert} heatmap overlay. "
            "Compare them and use the heatmap only as an attention hint. " + common
        )
    raise ValueError(condition)


def task_inputs(condition: str, row: Mapping[str, str], query_path: Path, reference_path: Path, artifact_root: Path):
    score = float(row["score_percentile_smoke"])
    if condition in {"score"}:
        images = [query_path]
    elif condition == "bbox":
        images = [artifact_root / row["bbox_overlay_path"]]
    elif condition == "heatmap":
        images = [artifact_root / row["heatmap_overlay_path"]]
    elif condition == "reference_score":
        images = [reference_path, query_path]
    elif condition == "reference_bbox":
        images = [reference_path, artifact_root / row["bbox_overlay_path"]]
    elif condition == "reference_heatmap":
        images = [reference_path, artifact_root / row["heatmap_overlay_path"]]
    else:
        raise ValueError(condition)
    for image in images:
        if not image.is_file(): raise FileNotFoundError(image)
    return images, prompt(condition, row["expert"], score)


def reusable(row: Mapping[str, object] | None, identity: Mapping[str, str]) -> bool:
    if not row or row.get("error") or not str(row.get("output", "")).strip(): return False
    return all(str(row.get(key, "")) == value for key, value in identity.items())


def evaluate(qa_rows, prediction_rows, *, model_key, model_id, condition, expert):
    pred = {str(row["id"]): row for row in prediction_rows}
    y_true, y_pred = [], []
    review = []
    for gt in qa_rows:
        item = pred.get(str(gt["id"]), {})
        parsed = parse_anomaly_output(str(item.get("output", "")))
        y_true.append(str(gt["answer"]))
        y_pred.append(parsed.anomaly if parsed.valid and parsed.anomaly else INVALID)
        review.append({
            "id": gt["id"], "asset_category": gt["asset_category"], "answer": gt["answer"],
            "parsed_prediction": parsed.anomaly if parsed.valid else INVALID,
            "valid": parsed.valid, "correct": parsed.valid and parsed.anomaly == gt["answer"],
            "model_key": model_key, "condition": condition, "expert": expert,
            "output": item.get("output", ""), "error": item.get("error", ""),
            "latency_seconds": item.get("latency_seconds", ""),
            "score_percentile_smoke": item.get("score_percentile_smoke", ""),
        })
    overall = {
        "evaluation_protocol": "evidence_ablation_smoke",
        "model_key": model_key, "model_id": model_id, "condition": condition, "expert": expert,
        **anomaly_classification_summary(y_true, y_pred),
    }
    groups = defaultdict(list)
    for gt in qa_rows: groups[str(gt["asset_category"])].append(gt)
    by_asset = []
    for asset, rows in sorted(groups.items()):
        ay_true, ay_pred = [], []
        for gt in rows:
            parsed = parse_anomaly_output(str(pred.get(str(gt["id"]), {}).get("output", "")))
            ay_true.append(str(gt["answer"]))
            ay_pred.append(parsed.anomaly if parsed.valid and parsed.anomaly else INVALID)
        by_asset.append({
            "model_key": model_key, "condition": condition, "expert": expert, "asset_category": asset,
            **anomaly_classification_summary(ay_true, ay_pred),
        })
    return overall, by_asset, review


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--unsup-zip", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, default=Path("/content/evidence_reference_cache"))
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY))
    parser.add_argument("--experts", nargs="+", required=True)
    parser.add_argument("--conditions", nargs="+", default=list(EVIDENCE_CONDITIONS))
    parser.add_argument("--checkpoint-every", type=int, default=1)
    args = parser.parse_args()
    unknown = set(args.conditions) - set(EVIDENCE_CONDITIONS)
    if unknown: parser.error(f"unknown conditions: {sorted(unknown)}")
    unknown_models = set(args.models) - set(MODEL_REGISTRY)
    if unknown_models: parser.error(f"unknown models: {sorted(unknown_models)}")

    qa_rows = read_jsonl(args.qa)
    evidence_rows = read_csv(args.evidence_manifest)
    evidence = {(row["query_id"], row["expert"]): row for row in evidence_rows}
    query_paths = workspace_queries(args.workspace_root)
    args.out_dir.mkdir(parents=True, exist_ok=True); args.image_cache.mkdir(parents=True, exist_ok=True)

    all_overall, all_asset, all_review = [], [], []
    status = []
    for model_key in args.models:
        spec = MODEL_REGISTRY[model_key]
        adapter = None
        try:
            print(f"=== load {model_key} ===")
            adapter = load_hf_adapter(spec)
            for expert in args.experts:
                for condition in args.conditions:
                    pred_path = args.out_dir / "predictions" / model_key / f"{condition}__{expert}.jsonl"
                    old_rows = read_jsonl(pred_path) if pred_path.is_file() else []
                    old = {str(row["id"]): row for row in old_rows}
                    outputs = []
                    for index, gt in enumerate(qa_rows, 1):
                        row = evidence[(str(gt["id"]), expert)]
                        reference_path = safe_extract_member(
                            args.unsup_zip, str(gt["reference_image"]), args.image_cache
                        )
                        identity = {
                            "condition": condition, "expert": expert,
                            "query_image": str(gt["image"]), "reference_image": str(gt["reference_image"]),
                            "score_percentile_smoke": str(row["score_percentile_smoke"]),
                        }
                        existing = old.get(str(gt["id"]))
                        if reusable(existing, identity):
                            outputs.append(existing); continue
                        images, prompt_text = task_inputs(
                            condition, row, query_paths[str(gt["id"])], reference_path, args.artifact_root
                        )
                        started = time.perf_counter(); output = ""; error = ""
                        try:
                            output = str(infer_hf(adapter, images, prompt_text))
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
                        result = {
                            "id": gt["id"], "model_key": model_key, "model_id": spec["model_id"],
                            **identity, "output": output, "error": error,
                            "latency_seconds": round(time.perf_counter() - started, 6),
                        }
                        outputs.append(result)
                        if index % args.checkpoint_every == 0 or index == len(qa_rows):
                            atomic_write_jsonl(pred_path, outputs)
                            print(f"{model_key}/{expert}/{condition}: {index}/{len(qa_rows)}")
                    atomic_write_jsonl(pred_path, outputs)
                    overall, by_asset, review = evaluate(
                        qa_rows, outputs, model_key=model_key, model_id=spec["model_id"],
                        condition=condition, expert=expert,
                    )
                    all_overall.append(overall); all_asset.extend(by_asset); all_review.extend(review)
                    status.append({
                        "model_key": model_key, "expert": expert, "condition": condition,
                        "status": "complete", "error_rows": sum(bool(row.get("error")) for row in outputs),
                    })
        finally:
            adapter = None; gc.collect()
            try:
                import torch
                if torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.ipc_collect()
            except Exception: pass

    def upsert(path: Path, new_rows, keys):
        old_rows = read_csv(path) if path.is_file() and path.stat().st_size else []
        replace = {tuple(str(row.get(key, "")) for key in keys) for row in new_rows}
        kept = [row for row in old_rows if tuple(str(row.get(key, "")) for key in keys) not in replace]
        write_csv_atomic(path, [*kept, *new_rows])

    upsert(args.out_dir / "leaderboard.csv", all_overall, ("model_key", "expert", "condition"))
    upsert(args.out_dir / "anomaly_by_asset.csv", all_asset, ("model_key", "expert", "condition", "asset_category"))
    upsert(args.out_dir / "prediction_review.csv", all_review, ("id", "model_key", "expert", "condition"))
    status_path = args.out_dir / "run_status.json"
    old_status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else []
    status_keys = {(row["model_key"], row["expert"], row["condition"]) for row in status}
    kept_status = [row for row in old_status if (row.get("model_key"), row.get("expert"), row.get("condition")) not in status_keys]
    status_path.write_text(json.dumps([*kept_status, *status], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
