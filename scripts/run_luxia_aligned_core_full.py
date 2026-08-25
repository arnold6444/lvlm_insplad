from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from insplad_benchmark.luxia_adapter import (
    LUXIA_CHAT_MODEL,
    LUXIA_DOCUMENT_AI_ENDPOINT,
    infer_luxia_pipeline,
)

MODEL_KEY = "luxia_document_ai_chat"
MODEL_ID = f"document-ai + {LUXIA_CHAT_MODEL}"
ROLE = "luxia_api_pipeline"

ANOMALY_PROMPT = (
    'Inspect the target power-line asset image. Classify it as normal or anomalous. '
    'Return only one JSON object exactly like {"anomaly":"normal"} or {"anomaly":"anomalous"}. '
    'Do not add explanation.'
)
ONE_SHOT_PROMPT = (
    'Image 1 is a NORMAL reference of the same asset category. Image 2 is the TARGET image. '
    'Compare the target with the normal reference and classify only the TARGET as normal or anomalous. '
    'Return only one JSON object exactly like {"anomaly":"normal"} or {"anomaly":"anomalous"}. '
    'Do not add explanation.'
)


def structured_prompt(row: dict) -> str:
    classes = ", ".join(row["valid_defect_classes"])
    return (
        "Inspect the target power-line asset. Predict both anomaly status and defect class. "
        f"Allowed defect_class values for this asset are: {classes}. "
        "Return only one JSON object with exactly two keys: "
        '{"anomaly":"normal|anomalous","defect_class":"<allowed class>"}. '
        'If normal, defect_class must be "normal". Do not add explanation.'
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reusable_prediction(row: dict | None) -> bool:
    return bool(row and not row.get("error") and str(row.get("output", "")).strip())


def safe_extract(zip_path: Path, member: str, out_root: Path, dataset_key: str) -> Path:
    out = out_root / dataset_key / member
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as archive:
        try:
            data = archive.read(member)
        except KeyError:
            normalized = member.replace("\\", "/").lstrip("/")
            matches = [
                name for name in archive.namelist()
                if name.replace("\\", "/").lstrip("/") == normalized
            ]
            if not matches:
                raise
            data = archive.read(matches[0])
    out.write_bytes(data)
    return out


def load_qa(qa_dir: Path, limit: int | None) -> dict[str, list[dict]]:
    qa = {
        "zero_shot": read_jsonl(qa_dir / "zero_shot.jsonl"),
        "one_shot": read_jsonl(qa_dir / "one_shot.jsonl"),
        "structured": read_jsonl(qa_dir / "structured.jsonl"),
    }
    if limit is not None:
        qa = {key: rows[:limit] for key, rows in qa.items()}
    counts = {key: len(rows) for key, rows in qa.items()}
    if limit is None and counts != {"zero_shot": 1378, "one_shot": 1378, "structured": 1378}:
        raise RuntimeError(f"Aligned Core QA must be 1378/1378/1378, got {counts}")
    if [r["id"] for r in qa["zero_shot"]] != [r["id"] for r in qa["one_shot"]]:
        raise RuntimeError("zero-shot and one-shot query IDs differ")
    return qa


def extract_images(qa_sets, unsup_zip: Path, sup_zip: Path, image_root: Path) -> dict[str, Path]:
    cache: dict[str, Path] = {}
    for task, rows in qa_sets.items():
        for row in rows:
            if task in {"zero_shot", "one_shot"}:
                cache[row["image"]] = safe_extract(unsup_zip, row["image"], image_root, "unsup")
                if row.get("reference_image"):
                    cache[row["reference_image"]] = safe_extract(
                        unsup_zip, row["reference_image"], image_root, "unsup"
                    )
            else:
                cache[row["image"]] = safe_extract(sup_zip, row["image"], image_root, "sup")
    return cache


def task_inputs(task: str, row: dict, cache: dict[str, Path]) -> tuple[list[Path], str]:
    if task == "zero_shot":
        return [cache[row["image"]]], ANOMALY_PROMPT
    if task == "one_shot":
        return [cache[row["reference_image"]], cache[row["image"]]], ONE_SHOT_PROMPT
    return [cache[row["image"]]], structured_prompt(row)


def call_with_retries(fn, attempts: int = 3):
    last_error = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
    raise last_error


def evaluate_predictions(qa_dir: Path, out_dir: Path) -> None:
    metrics_dir = out_dir / "metrics"
    tasks = {
        "zero_shot": ("anomaly", qa_dir / "zero_shot.jsonl", "zero_shot"),
        "one_shot": ("anomaly", qa_dir / "one_shot.jsonl", "one_shot"),
        "structured": ("explainability", qa_dir / "structured.jsonl", None),
    }
    for task_name, (task, qa, regime) in tasks.items():
        cmd = [
            sys.executable, str(ROOT / "scripts" / "evaluate.py"),
            "--task", task,
            "--qa", str(qa),
            "--predictions", str(out_dir / f"{task_name}.jsonl"),
            "--out-dir", str(metrics_dir),
        ]
        if regime:
            cmd += ["--regime", regime]
        subprocess.run(cmd, check=True)


def first_row(path: Path, regime: str | None = None) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if regime is None:
        return rows[0]
    return next(row for row in rows if row.get("regime") == regime)


def build_leaderboard(out_dir: Path) -> None:
    zero = first_row(out_dir / "metrics/anomaly_overall.csv", "zero_shot")
    one = first_row(out_dir / "metrics/anomaly_overall.csv", "one_shot")
    structured = first_row(out_dir / "metrics/explainability_overall.csv")
    row = {
        "model_key": MODEL_KEY,
        "model_id": MODEL_ID,
        "role": ROLE,
        "backend": "api_pipeline",
        "provider": "luxia",
        "zero_bacc": zero["balanced_accuracy"],
        "zero_f1": zero["f1"],
        "zero_invalid": zero["invalid_output_rate"],
        "one_bacc": one["balanced_accuracy"],
        "one_f1": one["f1"],
        "one_invalid": one["invalid_output_rate"],
        "defect_bacc": structured["defect_balanced_accuracy"],
        "defect_macro_f1": structured["defect_macro_f1"],
        "joint_accuracy": structured["joint_accuracy"],
        "false_defect_rate": structured["false_defect_rate"],
        "consistency_error": structured["consistency_error"],
        "structured_invalid": structured["invalid_output_rate"],
    }
    write_csv(out_dir / "leaderboard.csv", [row])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-dir", type=Path, required=True)
    parser.add_argument("--unsup-zip", type=Path, required=True)
    parser.add_argument("--sup-zip", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    api_key = os.environ.get("LUXIA_API_KEY", "")
    if not api_key:
        raise RuntimeError("LUXIA_API_KEY environment variable is required.")

    qa_sets = load_qa(args.qa_dir, args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_root = args.out_dir / "_images"
    path_cache = extract_images(qa_sets, args.unsup_zip, args.sup_zip, image_root)

    row_error_count = 0
    error_examples: list[dict] = []
    review_rows: list[dict] = []
    reused_predictions = 0
    rows_called_this_run = 0
    total_latency_s = 0.0

    for task, rows in qa_sets.items():
        pred_path = args.out_dir / f"{task}.jsonl"
        existing_rows = read_jsonl(pred_path) if pred_path.exists() else []
        existing_by_id = {row["id"]: row for row in existing_rows if row.get("id")}
        predictions: list[dict] = []

        for index, row in enumerate(rows, start=1):
            existing = existing_by_id.get(row["id"])
            if reusable_prediction(existing):
                prediction = existing
                output = str(prediction.get("output", ""))
                error = None
                latency_s = float(prediction.get("latency_s") or 0.0)
                reused_predictions += 1
                print(f"[{task}] {index}/{len(rows)} reuse")
            else:
                image_paths, prompt = task_inputs(task, row, path_cache)
                print(f"[{task}] {index}/{len(rows)} call")
                started = time.time()
                rows_called_this_run += 1
                try:
                    result = call_with_retries(
                        lambda: infer_luxia_pipeline(image_paths, prompt, api_key)
                    )
                    output = result["output"]
                    latency_s = float(result["latency_s"])
                    error = None
                except Exception as exc:
                    output = ""
                    latency_s = round(time.time() - started, 3)
                    error = f"{type(exc).__name__}: {exc}"
                total_latency_s += latency_s
                prediction = {
                    "id": row["id"],
                    "output": output,
                    "error": error,
                    "latency_s": round(latency_s, 3),
                    "model_key": MODEL_KEY,
                    "model_id": MODEL_ID,
                    "backend": "api_pipeline",
                    "provider": "luxia",
                }

            if error:
                row_error_count += 1
                if len(error_examples) < 10:
                    error_examples.append({"task": task, "id": row["id"], "error": error})

            predictions.append(prediction)
            review_rows.append({
                "model_key": MODEL_KEY,
                "model_id": MODEL_ID,
                "role": ROLE,
                "task": task,
                "id": row["id"],
                "asset_category": row.get("asset_category"),
                "ground_truth": json.dumps(row["answer"], ensure_ascii=False),
                "output": output,
                "error": error,
                "latency_s": round(latency_s, 3),
            })

            if index % args.checkpoint_every == 0:
                write_jsonl(pred_path, predictions)

        write_jsonl(pred_path, predictions)

    status = "completed" if row_error_count == 0 else "completed_with_row_errors"
    run_status = {
        "model_key": MODEL_KEY,
        "model_id": MODEL_ID,
        "backend": "api_pipeline",
        "provider": "luxia",
        "role": ROLE,
        "status": status,
        "row_error_count": row_error_count,
        "error_examples": error_examples,
        "reused_predictions": reused_predictions,
        "rows_called_this_run": rows_called_this_run,
        "total_latency_s": round(total_latency_s, 3),
    }
    (args.out_dir / "run_status.json").write_text(
        json.dumps(run_status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.out_dir / "prediction_review.csv", review_rows)

    evaluate_predictions(args.qa_dir, args.out_dir)
    build_leaderboard(args.out_dir)

    n_zero = len(qa_sets["zero_shot"])
    n_one = len(qa_sets["one_shot"])
    n_structured = len(qa_sets["structured"])
    metadata = {
        "provider": "luxia",
        "pipeline": "document_ai_plus_chat",
        "vision_endpoint": LUXIA_DOCUMENT_AI_ENDPOINT,
        "chat_model": LUXIA_CHAT_MODEL,
        "qa_counts": {key: len(value) for key, value in qa_sets.items()},
        "expected_vision_calls_without_resume": n_zero + 2 * n_one + n_structured,
        "expected_chat_calls_without_resume": n_zero + n_one + n_structured,
        "reused_predictions": reused_predictions,
        "rows_called_this_run": rows_called_this_run,
        "source_zip_paths": [str(args.unsup_zip), str(args.sup_zip)],
        "result_root": str(args.out_dir),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
