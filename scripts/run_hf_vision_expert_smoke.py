#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from insplad_benchmark.hf_adapters import infer_hf, load_hf_adapter
from insplad_benchmark.metrics import anomaly_classification_summary
from insplad_benchmark.parsing import parse_anomaly_output
from insplad_benchmark.vision_expert import read_manifest, resolve_artifact_path


MODEL_REGISTRY = {
    "qwen3_5_4b": {
        "backend": "hf",
        "provider": "huggingface",
        "model_id": "Qwen/Qwen3.5-4B",
        "trust_remote_code": False,
        "adapter": "pipeline",
        "role": "qwen_family",
    },
    "internvl3_5_8b": {
        "backend": "hf",
        "provider": "huggingface",
        "model_id": "OpenGVLab/InternVL3_5-8B",
        "trust_remote_code": True,
        "adapter": "internvl_chat",
        "max_total_tiles": 6,
        "role": "internvl_family",
    },
}
DEFAULT_MODELS = tuple(MODEL_REGISTRY)
CONDITIONS = ("roi", "roi_plus_reference")
INVALID = "<invalid>"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract_member(zip_path: Path, member: str, image_root: Path) -> Path:
    normalized = member.replace("\\", "/").lstrip("/")
    out = image_root / normalized
    if out.is_file():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as archive:
        try:
            data = archive.read(member)
        except KeyError:
            matches = [
                name
                for name in archive.namelist()
                if name.replace("\\", "/").lstrip("/") == normalized
            ]
            if len(matches) != 1:
                raise FileNotFoundError(f"ZIP member not found or ambiguous: {member}")
            data = archive.read(matches[0])
    out.write_bytes(data)
    return out


def prompt_for_condition(condition: str, expert: str) -> str:
    expert_label = expert.replace("_", "-")
    if condition == "roi":
        return (
            "Image 1 is the complete TARGET power-line asset image. "
            f"Image 2 is a candidate close-up selected by the {expert_label} vision expert. "
            "The close-up is only an attention hint and can be normal, so do not assume an anomaly merely because it was selected. "
            "Use both images to classify only the TARGET as normal or anomalous. "
            'Return only one JSON object exactly like {"anomaly":"normal"} or '
            '{"anomaly":"anomalous"}. Do not add explanation.'
        )
    if condition == "roi_plus_reference":
        return (
            "Image 1 is a NORMAL reference from the same asset category. "
            "Image 2 is the complete TARGET power-line asset image. "
            f"Image 3 is a candidate close-up selected from the TARGET by the {expert_label} vision expert. "
            "The close-up is only an attention hint and can be normal. Compare the target with the normal reference, "
            "inspect the close-up, and classify only the TARGET as normal or anomalous. "
            'Return only one JSON object exactly like {"anomaly":"normal"} or '
            '{"anomaly":"anomalous"}. Do not add explanation.'
        )
    raise ValueError(f"unsupported condition: {condition}")


def task_inputs(
    *,
    condition: str,
    query_path: Path,
    roi_path: Path,
    reference_path: Path | None,
    expert: str,
) -> tuple[list[Path], str]:
    if condition == "roi":
        return [query_path, roi_path], prompt_for_condition(condition, expert)
    if reference_path is None:
        raise ValueError("roi_plus_reference requires a reference image")
    return [reference_path, query_path, roi_path], prompt_for_condition(condition, expert)


def reusable_prediction(row: Mapping[str, object] | None, expected: Mapping[str, str]) -> bool:
    if not row or not str(row.get("output", "")).strip() or row.get("error"):
        return False
    return all(str(row.get(key, "")) == value for key, value in expected.items())


def call_with_retries(function, attempts: int) -> str:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return str(function())
        except Exception as exc:  # pragma: no cover - model/runtime dependent
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def _resolve_models(requested: Sequence[str]) -> list[str]:
    models = list(requested)
    unknown = sorted(set(models) - set(MODEL_REGISTRY))
    if unknown:
        raise ValueError(f"unknown/unsupported screening models: {unknown}")
    if len(models) != len(set(models)):
        raise ValueError("model keys must be unique")
    return models


def _load_roi_index(
    manifest_path: Path,
    *,
    artifact_root: Path | None,
    experts: Sequence[str],
    qa_by_id: Mapping[str, Mapping[str, object]],
) -> dict[tuple[str, str], dict[str, object]]:
    selected_experts = set(experts)
    index: dict[tuple[str, str], dict[str, object]] = {}
    for raw in read_manifest(manifest_path):
        expert = str(raw.get("expert", ""))
        query_id = str(raw.get("query_id", ""))
        if expert not in selected_experts or query_id not in qa_by_id:
            continue
        key = (query_id, expert)
        if key in index:
            raise ValueError(f"duplicate ROI manifest row: {key}")
        if str(raw.get("query_image", "")) != str(qa_by_id[query_id]["image"]):
            raise ValueError(f"ROI/query mismatch for {key}")
        roi_path = resolve_artifact_path(manifest_path, str(raw.get("roi_path", "")), artifact_root)
        if not roi_path.is_file():
            raise FileNotFoundError(f"ROI file missing for {key}: {roi_path}")
        row: dict[str, object] = dict(raw)
        row["resolved_roi_path"] = roi_path
        index[key] = row

    expected = {(query_id, expert) for query_id in qa_by_id for expert in selected_experts}
    missing = sorted(expected - set(index))
    if missing:
        raise ValueError(f"ROI manifest is incomplete; first missing rows: {missing[:20]}")
    return index


def _prediction_file(out_dir: Path, model_key: str, condition: str, expert: str) -> Path:
    return out_dir / "predictions" / model_key / f"{condition}__{expert}.jsonl"


def _evaluate_rows(
    qa_rows: Sequence[Mapping[str, object]],
    prediction_rows: Sequence[Mapping[str, object]],
    *,
    model_key: str,
    model_id: str,
    condition: str,
    expert: str,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    predictions = {str(row["id"]): row for row in prediction_rows}

    def summarize(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
        y_true: list[str] = []
        y_pred: list[str] = []
        for gt in rows:
            y_true.append(str(gt["answer"]))
            output = str(predictions.get(str(gt["id"]), {}).get("output", ""))
            parsed = parse_anomaly_output(output)
            y_pred.append(parsed.anomaly if parsed.valid and parsed.anomaly else INVALID)
        return dict(anomaly_classification_summary(y_true, y_pred))

    overall: dict[str, object] = {
        "evaluation_protocol": "vision_expert_screening_smoke",
        "model_key": model_key,
        "model_id": model_id,
        "condition": condition,
        "expert": expert,
        **summarize(qa_rows),
    }

    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in qa_rows:
        groups[str(row["asset_category"])].append(row)
    by_asset = []
    for asset, rows in sorted(groups.items()):
        by_asset.append(
            {
                "evaluation_protocol": "vision_expert_screening_smoke",
                "model_key": model_key,
                "model_id": model_id,
                "condition": condition,
                "expert": expert,
                "asset_category": asset,
                **summarize(rows),
            }
        )

    review: list[dict[str, object]] = []
    for gt in qa_rows:
        pred = predictions.get(str(gt["id"]), {})
        parsed = parse_anomaly_output(str(pred.get("output", "")))
        review.append(
            {
                "id": gt["id"],
                "asset_category": gt["asset_category"],
                "answer": gt["answer"],
                "parsed_prediction": parsed.anomaly if parsed.valid else INVALID,
                "valid": parsed.valid,
                "correct": parsed.valid and parsed.anomaly == gt["answer"],
                "model_key": model_key,
                "condition": condition,
                "expert": expert,
                "query_image": gt["image"],
                "reference_image": gt.get("reference_image", ""),
                "roi_path": pred.get("roi_path", ""),
                "output": pred.get("output", ""),
                "error": pred.get("error", ""),
                "latency_seconds": pred.get("latency_seconds", ""),
            }
        )
    return overall, by_asset, review


def _free_adapter(adapter: object | None) -> None:
    if adapter is not None:
        del adapter
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the fixed InsPLAD smoke IDs with vision-expert ROI evidence."
    )
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--roi-manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--unsup-zip", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, default=Path("/content/vision_expert_image_cache"))
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--experts", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--continue-on-model-error", action="store_true")
    args = parser.parse_args()

    if args.checkpoint_every <= 0 or args.attempts <= 0:
        parser.error("checkpoint-every and attempts must be positive")
    models = _resolve_models(args.models)
    experts = list(dict.fromkeys(args.experts))
    if len(experts) != len(args.experts):
        parser.error("expert keys must be unique")

    qa_rows = read_jsonl(args.qa)
    if not 20 <= len(qa_rows) <= 100:
        raise RuntimeError(f"screening QA must contain 20-100 rows, got {len(qa_rows)}")
    qa_by_id = {str(row.get("id", "")): row for row in qa_rows}
    if "" in qa_by_id or len(qa_by_id) != len(qa_rows):
        raise RuntimeError("QA IDs must be present and unique")
    required_qa = ("image", "reference_image", "answer", "asset_category")
    for row in qa_rows:
        missing = [field for field in required_qa if not row.get(field)]
        if missing:
            raise RuntimeError(f"QA row {row.get('id')} missing fields: {missing}")

    roi_index = _load_roi_index(
        args.roi_manifest,
        artifact_root=args.artifact_root,
        experts=experts,
        qa_by_id=qa_by_id,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.image_cache.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "schema_version": 1,
        "evaluation_protocol": "vision_expert_screening_smoke",
        "qa_path": str(args.qa.resolve()),
        "qa_sha256": sha256_file(args.qa),
        "qa_count": len(qa_rows),
        "roi_manifest_path": str(args.roi_manifest.resolve()),
        "roi_manifest_sha256": sha256_file(args.roi_manifest),
        "artifact_root": str(args.artifact_root.resolve()) if args.artifact_root else None,
        "condition": args.condition,
        "experts": experts,
        "models": models,
        "input_order": (
            ["query", "roi"]
            if args.condition == "roi"
            else ["normal_reference", "query", "roi"]
        ),
        "selection_warning": "This smoke set is for method selection, not the final paper score.",
    }
    metadata_path = args.out_dir / f"run_metadata__{args.condition}.json"
    metadata_path.write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    all_leaderboard: list[dict[str, object]] = []
    all_by_asset: list[dict[str, object]] = []
    all_review: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []

    for model_key in models:
        spec = MODEL_REGISTRY[model_key]
        adapter = None
        try:
            print(f"\n=== Loading {model_key}: {spec['model_id']} ===")
            adapter = load_hf_adapter(spec)
            for expert in experts:
                pred_path = _prediction_file(args.out_dir, model_key, args.condition, expert)
                existing_rows = read_jsonl(pred_path) if pred_path.is_file() else []
                existing = {str(row.get("id", "")): row for row in existing_rows}
                outputs: list[dict[str, object]] = []

                for index, gt in enumerate(qa_rows, start=1):
                    query_id = str(gt["id"])
                    roi_row = roi_index[(query_id, expert)]
                    roi_path = Path(str(roi_row["resolved_roi_path"]))
                    query_path = safe_extract_member(args.unsup_zip, str(gt["image"]), args.image_cache)
                    reference_path = (
                        safe_extract_member(args.unsup_zip, str(gt["reference_image"]), args.image_cache)
                        if args.condition == "roi_plus_reference"
                        else None
                    )
                    expected_identity = {
                        "condition": args.condition,
                        "expert": expert,
                        "query_image": str(gt["image"]),
                        "reference_image": str(gt.get("reference_image", ""))
                        if args.condition == "roi_plus_reference"
                        else "",
                        "roi_path": str(roi_path.resolve()),
                    }
                    if reusable_prediction(existing.get(query_id), expected_identity):
                        outputs.append(existing[query_id])
                        continue

                    images, prompt = task_inputs(
                        condition=args.condition,
                        query_path=query_path,
                        roi_path=roi_path,
                        reference_path=reference_path,
                        expert=expert,
                    )
                    started = time.perf_counter()
                    output = ""
                    error = ""
                    try:
                        output = call_with_retries(
                            lambda: infer_hf(adapter, images, prompt),
                            attempts=args.attempts,
                        )
                    except Exception as exc:  # pragma: no cover - model/runtime dependent
                        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"
                    row: dict[str, object] = {
                        "id": query_id,
                        "model_key": model_key,
                        "model_id": spec["model_id"],
                        "condition": args.condition,
                        "expert": expert,
                        "query_image": str(gt["image"]),
                        "reference_image": str(gt.get("reference_image", ""))
                        if args.condition == "roi_plus_reference"
                        else "",
                        "roi_path": str(roi_path.resolve()),
                        "roi_anomaly_score_top1pct": roi_row.get("anomaly_score_top1pct", ""),
                        "output": output,
                        "error": error,
                        "latency_seconds": round(time.perf_counter() - started, 6),
                    }
                    outputs.append(row)
                    if index % args.checkpoint_every == 0 or index == len(qa_rows):
                        write_jsonl_atomic(pred_path, outputs)
                        print(
                            f"{model_key} / {expert} / {args.condition}: "
                            f"{index}/{len(qa_rows)} checkpoint"
                        )

                write_jsonl_atomic(pred_path, outputs)
                overall, by_asset, review = _evaluate_rows(
                    qa_rows,
                    outputs,
                    model_key=model_key,
                    model_id=str(spec["model_id"]),
                    condition=args.condition,
                    expert=expert,
                )
                all_leaderboard.append(overall)
                all_by_asset.extend(by_asset)
                all_review.extend(review)
                status_rows.append(
                    {
                        "model_key": model_key,
                        "model_id": spec["model_id"],
                        "condition": args.condition,
                        "expert": expert,
                        "completed_rows": len(outputs),
                        "error_rows": sum(bool(row.get("error")) for row in outputs),
                        "prediction_file": str(pred_path.resolve()),
                        "status": "complete",
                    }
                )
        except Exception as exc:  # pragma: no cover - model/runtime dependent
            status_rows.append(
                {
                    "model_key": model_key,
                    "model_id": spec["model_id"],
                    "condition": args.condition,
                    "expert": "*",
                    "completed_rows": 0,
                    "error_rows": len(qa_rows) * len(experts),
                    "prediction_file": "",
                    "status": "model_error",
                    "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}",
                }
            )
            if not args.continue_on_model_error:
                raise
        finally:
            adapter = None
            _free_adapter(None)

    # This script may be invoked once per condition. Preserve additive rows from
    # the other condition while replacing exact model/condition/expert keys.
    def upsert_csv(path: Path, new_rows: list[dict[str, object]], keys: tuple[str, ...]) -> None:
        old_rows: list[dict[str, object]] = []
        if path.is_file() and path.stat().st_size:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                old_rows = list(csv.DictReader(handle))
        replace = {tuple(str(row.get(key, "")) for key in keys) for row in new_rows}
        kept = [
            row
            for row in old_rows
            if tuple(str(row.get(key, "")) for key in keys) not in replace
        ]
        write_csv(path, [*kept, *new_rows])

    upsert_csv(
        args.out_dir / "leaderboard.csv",
        all_leaderboard,
        ("model_key", "condition", "expert"),
    )
    upsert_csv(
        args.out_dir / "anomaly_by_asset.csv",
        all_by_asset,
        ("model_key", "condition", "expert", "asset_category"),
    )
    upsert_csv(
        args.out_dir / "prediction_review.csv",
        all_review,
        ("id", "model_key", "condition", "expert"),
    )
    status_path = args.out_dir / f"run_status__{args.condition}.json"
    status_path.write_text(json.dumps(status_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if all_leaderboard:
        print("\nCompleted screening rows:")
        for row in all_leaderboard:
            print(
                f"- {row['model_key']} / {row['expert']} / {row['condition']}: "
                f"F1={float(row['f1']):.4f}, BAcc={float(row['balanced_accuracy']):.4f}, "
                f"invalid={float(row['invalid_output_rate']):.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
