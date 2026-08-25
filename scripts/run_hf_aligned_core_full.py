from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from tqdm.auto import tqdm

from insplad_benchmark.hf_adapters import infer_hf, load_hf_adapter

MODEL_REGISTRY = {
    "qwen3_5_4b": {
        "backend": "hf", "provider": "huggingface",
        "model_id": "Qwen/Qwen3.5-4B",
        "trust_remote_code": False, "adapter": "pipeline",
        "role": "qwen_family",
    },
    "internvl3_5_8b": {
        "backend": "hf", "provider": "huggingface",
        "model_id": "OpenGVLab/InternVL3_5-8B",
        "trust_remote_code": True, "adapter": "internvl_chat",
        "max_total_tiles": 6, "role": "internvl_family",
    },
    "minicpm_v_4_5": {
        "backend": "hf", "provider": "huggingface",
        "model_id": "openbmb/MiniCPM-V-4_5",
        "trust_remote_code": True, "adapter": "minicpm_chat",
        "role": "minicpm_family",
    },
    "llava_onevision_7b": {
        "backend": "hf", "provider": "huggingface",
        "model_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "trust_remote_code": False, "adapter": "llava_processor",
        "role": "llava_family",
    },
}

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


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_extract_member(zip_path: Path, member: str, image_root: Path, dataset_key: str) -> Path:
    out = image_root / dataset_key / member
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as zf:
        try:
            data = zf.read(member)
        except KeyError:
            normalized = member.replace("\\", "/").lstrip("/")
            matches = [name for name in zf.namelist() if name.replace("\\", "/").lstrip("/") == normalized]
            if not matches:
                raise
            data = zf.read(matches[0])
    out.write_bytes(data)
    return out


def structured_prompt(row: dict) -> str:
    classes = ", ".join(row["valid_defect_classes"])
    return (
        "Inspect the target power-line asset. Predict both anomaly status and defect class. "
        f"Allowed defect_class values for this asset are: {classes}. "
        "Return only one JSON object with exactly two keys: "
        '{"anomaly":"normal|anomalous","defect_class":"<allowed class>"}. '
        'If normal, defect_class must be "normal". Do not add explanation.'
    )


def reusable_prediction(row: dict | None) -> bool:
    return bool(row and row.get("output")) and not row.get("error")


def call_with_retries(fn, attempts: int = 3):
    last_error = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if attempt + 1 == attempts:
                break
            time.sleep(2 ** attempt)
    raise last_error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-dir", type=Path, required=True)
    parser.add_argument("--unsup-zip", type=Path, required=True)
    parser.add_argument("--sup-zip", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("/content/insplad_aligned_core_images"))
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY))
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()

    unknown = sorted(set(args.models) - set(MODEL_REGISTRY))
    if unknown:
        parser.error(f"unknown models: {unknown}")

    qa = {
        "zero_shot": read_jsonl(args.qa_dir / "zero_shot.jsonl"),
        "one_shot": read_jsonl(args.qa_dir / "one_shot.jsonl"),
        "structured": read_jsonl(args.qa_dir / "structured.jsonl"),
    }
    if not all(len(rows) == 1378 for rows in qa.values()):
        raise RuntimeError(f"Aligned Core QA must be 1378/1378/1378, got { {k: len(v) for k, v in qa.items()} }")
    if [r["id"] for r in qa["zero_shot"]] != [r["id"] for r in qa["one_shot"]]:
        raise RuntimeError("zero-shot and one-shot query IDs differ")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.image_root.mkdir(parents=True, exist_ok=True)

    path_cache: dict[str, Path] = {}
    for task, rows in qa.items():
        for row in rows:
            if task in {"zero_shot", "one_shot"}:
                if row["image"] not in path_cache:
                    path_cache[row["image"]] = safe_extract_member(args.unsup_zip, row["image"], args.image_root, "unsup")
                ref = row.get("reference_image")
                if ref and ref not in path_cache:
                    path_cache[ref] = safe_extract_member(args.unsup_zip, ref, args.image_root, "unsup")
            elif row["image"] not in path_cache:
                path_cache[row["image"]] = safe_extract_member(args.sup_zip, row["image"], args.image_root, "sup")

    def task_inputs(task: str, row: dict):
        if task == "zero_shot":
            return [path_cache[row["image"]]], ANOMALY_PROMPT
        if task == "one_shot":
            return [path_cache[row["reference_image"]], path_cache[row["image"]]], ONE_SHOT_PROMPT
        return [path_cache[row["image"]]], structured_prompt(row)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    if vram_gb < 20:
        raise RuntimeError(f"L4/A100-class GPU required; VRAM={vram_gb:.1f} GB")

    run_status = []
    for model_key in args.models:
        spec = MODEL_REGISTRY[model_key]
        adapter = None
        print(f"\n=== {model_key} | {spec['model_id']} ===")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            adapter = load_hf_adapter(spec)
            reused = called = row_errors = 0
            error_examples: list[dict] = []

            for task, gt_rows in qa.items():
                pred_path = args.out_dir / model_key / f"{task}.jsonl"
                existing = read_jsonl(pred_path) if pred_path.is_file() else []
                existing_by_id = {r["id"]: r for r in existing}
                predictions = []

                for index, gt in enumerate(tqdm(gt_rows, desc=f"{model_key}:{task}"), 1):
                    old = existing_by_id.get(gt["id"])
                    if reusable_prediction(old):
                        predictions.append(old)
                        reused += 1
                    else:
                        image_paths, prompt = task_inputs(task, gt)
                        started = time.time()
                        try:
                            output = call_with_retries(lambda: infer_hf(adapter, image_paths, prompt))
                            error = None
                        except Exception as exc:
                            output = ""
                            error = f"{type(exc).__name__}: {exc}"
                            row_errors += 1
                            if len(error_examples) < 10:
                                error_examples.append({"task": task, "id": gt["id"], "error": error})
                        predictions.append({
                            "id": gt["id"], "output": output, "error": error,
                            "latency_s": round(time.time() - started, 3),
                            "model_key": model_key, "model_id": spec["model_id"],
                            "backend": "hf", "provider": "huggingface",
                        })
                        called += 1

                    if index % args.checkpoint_every == 0:
                        write_jsonl(pred_path, predictions)
                write_jsonl(pred_path, predictions)

            run_status.append({
                "model_key": model_key, "model_id": spec["model_id"],
                "status": "completed" if row_errors == 0 else "completed_with_row_errors",
                "row_error_count": row_errors, "error_examples": error_examples,
                "reused_predictions": reused, "rows_called_this_run": called,
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            })
        except Exception as exc:
            traceback.print_exc()
            run_status.append({
                "model_key": model_key, "model_id": spec["model_id"],
                "status": "load_or_run_failed", "error": f"{type(exc).__name__}: {exc}",
            })
        finally:
            if adapter is not None:
                del adapter
            gc.collect()
            torch.cuda.empty_cache()
        (args.out_dir / "run_status.json").write_text(
            json.dumps(run_status, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    task_eval = {
        "zero_shot": ("anomaly", args.qa_dir / "zero_shot.jsonl", "zero_shot"),
        "one_shot": ("anomaly", args.qa_dir / "one_shot.jsonl", "one_shot"),
        "structured": ("explainability", args.qa_dir / "structured.jsonl", None),
    }
    leaderboard = []
    for model_key, spec in MODEL_REGISTRY.items():
        model_dir = args.out_dir / model_key
        ready = True
        for task, gt_rows in qa.items():
            pred_path = model_dir / f"{task}.jsonl"
            if not pred_path.is_file():
                ready = False
                break
            by_id = {r["id"]: r for r in read_jsonl(pred_path)}
            if len(by_id) != 1378 or not all(reusable_prediction(by_id.get(gt["id"])) for gt in gt_rows):
                ready = False
                break
        if not ready:
            continue

        metrics_dir = model_dir / "metrics"
        if metrics_dir.exists():
            import shutil
            shutil.rmtree(metrics_dir)
        for task, (eval_task, qa_path, regime) in task_eval.items():
            cmd = [
                sys.executable, "scripts/evaluate.py",
                "--task", eval_task, "--qa", str(qa_path),
                "--predictions", str(model_dir / f"{task}.jsonl"),
                "--out-dir", str(metrics_dir),
            ]
            if regime:
                cmd += ["--regime", regime]
            subprocess.run(cmd, check=True)

        anomaly = pd.read_csv(metrics_dir / "anomaly_overall.csv")
        explain = pd.read_csv(metrics_dir / "explainability_overall.csv").iloc[0]
        zero = anomaly.loc[anomaly["regime"] == "zero_shot"].iloc[0]
        one = anomaly.loc[anomaly["regime"] == "one_shot"].iloc[0]
        leaderboard.append({
            "model_key": model_key, "model_id": spec["model_id"],
            "zero_bacc": zero["balanced_accuracy"], "zero_f1": zero["f1"],
            "one_bacc": one["balanced_accuracy"], "one_f1": one["f1"],
            "defect_bacc": explain["defect_balanced_accuracy"],
            "defect_macro_f1": explain["defect_macro_f1"],
            "joint_accuracy": explain["joint_accuracy"],
            "false_defect_rate": explain["false_defect_rate"],
            "consistency_error": explain["consistency_error"],
            "structured_invalid": explain["invalid_output_rate"],
        })

    pd.DataFrame(leaderboard).to_csv(args.out_dir / "leaderboard.csv", index=False)
    metadata = {
        "dataset": "InsPLAD Aligned Core", "aligned_core_count": 1378,
        "same_query_across_tasks": True, "models": MODEL_REGISTRY,
        "gpu": props.name, "vram_gb": round(vram_gb, 2),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(leaderboard))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
