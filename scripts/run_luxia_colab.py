from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import torch
from tqdm.auto import tqdm

from insplad_benchmark.hf_adapters import infer_hf, load_hf_adapter
from insplad_benchmark.luxia_api import DEFAULT_LUXIA_CHAT_URL, infer_luxia_external

try:
    from google.colab import files, userdata
except ImportError as exc:  # pragma: no cover - Colab-only runner
    raise RuntimeError("scripts/run_luxia_colab.py is intended for Google Colab.") from exc


def get_secret(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    try:
        return userdata.get(name)
    except Exception:
        return None


def read_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_extract_member(zip_path: Path, member: str, dataset_key: str, image_root: Path) -> Path:
    out = image_root / dataset_key / member
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as archive:
        try:
            data = archive.read(member)
        except KeyError:
            normalized = member.replace("\\", "/").lstrip("/")
            matches = [
                name
                for name in archive.namelist()
                if name.replace("\\", "/").lstrip("/") == normalized
            ]
            if not matches:
                raise
            data = archive.read(matches[0])
    out.write_bytes(data)
    return out


MODEL_REGISTRY = {
    "gpt_5_3_chat_luxia": {
        "backend": "api",
        "provider": "luxia_external",
        "vendor": "openai",
        "model_id": "gpt-5.3-chat-latest",
        "secret_name": "LUXIA_API_KEY",
    },
    "gemini_3_flash_luxia": {
        "backend": "api",
        "provider": "luxia_external",
        "vendor": "gemini",
        "model_id": "gemini-3-flash-preview",
        "secret_name": "LUXIA_API_KEY",
    },
    "qwen3_vl_8b": {
        "backend": "hf",
        "provider": "huggingface",
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "trust_remote_code": False,
        "adapter": "pipeline",
    },
    "internvl3_5_8b": {
        "backend": "hf",
        "provider": "huggingface",
        "model_id": "OpenGVLab/InternVL3_5-8B",
        "trust_remote_code": True,
        "adapter": "internvl_chat",
        "max_num_tiles": 12,
    },
    "minicpm_v_4_5": {
        "backend": "hf",
        "provider": "huggingface",
        "model_id": "openbmb/MiniCPM-V-4_5",
        "trust_remote_code": True,
        "adapter": "minicpm_chat",
    },
    "llava_onevision_7b": {
        "backend": "hf",
        "provider": "huggingface",
        "model_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "trust_remote_code": False,
        "adapter": "pipeline",
    },
}
API_MODELS_TO_RUN = ["gpt_5_3_chat_luxia", "gemini_3_flash_luxia"]
HF_MODELS_TO_RUN = ["qwen3_vl_8b", "internvl3_5_8b", "minicpm_v_4_5", "llava_onevision_7b"]
ALLOW_HF_BELOW_20GB = False
SMOKE_LIMIT = 20

ANOMALY_PROMPT = (
    "Inspect the target power-line asset image. Classify it as normal or anomalous. "
    'Return only one JSON object exactly like {"anomaly":"normal"} or '
    '{"anomaly":"anomalous"}. Do not add explanation.'
)
ONE_SHOT_PROMPT = (
    "Image 1 is a NORMAL reference of the same asset category. Image 2 is the TARGET image. "
    "Compare the target with the normal reference and classify only the TARGET as normal or anomalous. "
    'Return only one JSON object exactly like {"anomaly":"normal"} or '
    '{"anomaly":"anomalous"}. Do not add explanation.'
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


def call_with_retries(fn, attempts: int = 4):
    last_error = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - benchmark should record provider errors
            last_error = exc
            if attempt + 1 == attempts:
                break
            time.sleep(2**attempt)
    raise last_error


def main() -> None:
    repo_dir = Path(__file__).resolve().parents[1]
    os.chdir(repo_dir)
    benchmark_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    luxia_api_key = get_secret("LUXIA_API_KEY")
    luxia_chat_url = os.environ.get("LUXIA_EXTERNAL_CHAT_URL", DEFAULT_LUXIA_CHAT_URL)
    print("LUXIA_API_KEY configured:", bool(luxia_api_key))
    print("Luxia chat endpoint:", luxia_chat_url)

    colab_root = Path("/content")
    unsup_zip = colab_root / "unsupervised_anomaly_detection.zip"
    sup_zip = colab_root / "supervised_fault_classification.zip"
    missing = [str(path) for path in (unsup_zip, sup_zip) if not path.is_file()]
    if missing:
        uploaded = sorted(path.name for path in colab_root.glob("*.zip"))
        raise FileNotFoundError(
            "Upload both source ZIPs into /content before running. "
            f"Missing: {missing}. ZIPs currently in /content: {uploaded}"
        )

    qa_sets = {
        "zero_shot": read_jsonl("samples/anomaly_zero_shot_20.jsonl")[:SMOKE_LIMIT],
        "one_shot": read_jsonl("samples/anomaly_one_shot_20.jsonl")[:SMOKE_LIMIT],
        "structured": read_jsonl("samples/defect_20.jsonl")[:SMOKE_LIMIT],
    }
    image_root = Path("/content/insplad_smoke_images")
    image_root.mkdir(parents=True, exist_ok=True)
    path_cache: dict[str, Path] = {}
    for task, rows in qa_sets.items():
        for row in rows:
            if task in {"zero_shot", "one_shot"}:
                path_cache[row["image"]] = safe_extract_member(unsup_zip, row["image"], "unsup", image_root)
                if row.get("reference_image"):
                    path_cache[row["reference_image"]] = safe_extract_member(
                        unsup_zip, row["reference_image"], "unsup", image_root
                    )
            else:
                path_cache[row["image"]] = safe_extract_member(sup_zip, row["image"], "sup", image_root)
    print("QA counts:", {key: len(rows) for key, rows in qa_sets.items()})
    print("extracted unique images:", len(path_cache))

    def task_inputs(task: str, row: dict) -> tuple[list[Path], str]:
        if task == "zero_shot":
            return [path_cache[row["image"]]], ANOMALY_PROMPT
        if task == "one_shot":
            return [path_cache[row["reference_image"]], path_cache[row["image"]]], ONE_SHOT_PROMPT
        return [path_cache[row["image"]]], structured_prompt(row)

    def infer_api(spec: dict, image_paths: list[Path], prompt: str) -> str:
        return infer_luxia_external(
            api_key=luxia_api_key or "",
            model_id=spec["model_id"],
            image_paths=image_paths,
            prompt=prompt,
            endpoint=luxia_chat_url,
            max_completion_tokens=80,
        )

    cuda_available = torch.cuda.is_available()
    gpu_name = None
    vram_gb = 0.0
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        vram_gb = props.total_memory / 1024**3
    hf_ready = cuda_available and (vram_gb >= 20 or ALLOW_HF_BELOW_20GB)
    print("CUDA available:", cuda_available)
    print("GPU:", gpu_name)
    print(f"VRAM: {vram_gb:.1f} GB")
    print("HF direct inference ready:", hf_ready)

    api_probe_status: dict[str, dict] = {}
    if luxia_api_key:
        probe_images, probe_prompt = task_inputs("zero_shot", qa_sets["zero_shot"][0])
        for model_key in API_MODELS_TO_RUN:
            spec = MODEL_REGISTRY[model_key]
            try:
                probe_output = call_with_retries(
                    lambda _spec=spec: infer_api(_spec, probe_images, probe_prompt), attempts=2
                )
                api_probe_status[model_key] = {"ok": True, "output": probe_output}
                print("Luxia probe OK:", model_key, "=>", probe_output[:200])
            except Exception as exc:  # noqa: BLE001
                api_probe_status[model_key] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print("Luxia probe FAILED:", model_key, "=>", api_probe_status[model_key]["error"])
    else:
        print("Luxia probe skipped: LUXIA_API_KEY is missing.")

    run_base = Path("/content/lvlm_insplad_results")
    run_root = run_base / time.strftime("hybrid_smoke_%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    run_status: list[dict] = []

    def run_rows(model_key: str, spec: dict, infer_fn) -> None:
        model_dir = run_root / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        for task, rows in qa_sets.items():
            predictions = []
            for row in tqdm(rows, desc=f"{model_key}:{task}"):
                image_paths, prompt = task_inputs(task, row)
                started = time.time()
                try:
                    output = call_with_retries(lambda: infer_fn(image_paths, prompt))
                    error = None
                except Exception as exc:  # noqa: BLE001
                    output = ""
                    error = f"{type(exc).__name__}: {exc}"
                predictions.append(
                    {
                        "id": row["id"],
                        "output": output,
                        "error": error,
                        "latency_s": round(time.time() - started, 3),
                        "model_key": model_key,
                        "model_id": spec["model_id"],
                        "backend": spec["backend"],
                        "provider": spec["provider"],
                        "vendor": spec.get("vendor"),
                    }
                )
                write_jsonl(model_dir / f"{task}.jsonl", predictions)

    for model_key in API_MODELS_TO_RUN:
        spec = MODEL_REGISTRY[model_key]
        if not luxia_api_key:
            run_status.append(
                {
                    "model_key": model_key,
                    "model_id": spec["model_id"],
                    "backend": "api",
                    "provider": "luxia_external",
                    "status": "skipped_missing_api_key",
                }
            )
            continue
        probe = api_probe_status.get(model_key)
        if not probe or not probe.get("ok"):
            run_status.append(
                {
                    "model_key": model_key,
                    "model_id": spec["model_id"],
                    "backend": "api",
                    "provider": "luxia_external",
                    "status": "skipped_api_probe_failed",
                    "error": (probe or {}).get("error", "probe did not run"),
                }
            )
            continue
        print("\n=== LUXIA API", model_key, spec["model_id"], "===")
        run_rows(model_key, spec, lambda images, prompt, _spec=spec: infer_api(_spec, images, prompt))
        run_status.append(
            {
                "model_key": model_key,
                "model_id": spec["model_id"],
                "backend": "api",
                "provider": "luxia_external",
                "vendor": spec.get("vendor"),
                "status": "completed",
            }
        )

    if not hf_ready:
        for model_key in HF_MODELS_TO_RUN:
            spec = MODEL_REGISTRY[model_key]
            run_status.append(
                {
                    "model_key": model_key,
                    "model_id": spec["model_id"],
                    "backend": "hf",
                    "provider": "huggingface",
                    "status": "skipped_insufficient_gpu",
                    "gpu": gpu_name,
                    "vram_gb": round(vram_gb, 2),
                }
            )
    else:
        for model_key in HF_MODELS_TO_RUN:
            spec = MODEL_REGISTRY[model_key]
            print("\n=== HF", model_key, spec["model_id"], "===")
            adapter = None
            try:
                adapter = load_hf_adapter(spec)
                run_rows(
                    model_key,
                    spec,
                    lambda images, prompt, _adapter=adapter: infer_hf(_adapter, images, prompt),
                )
                run_status.append(
                    {
                        "model_key": model_key,
                        "model_id": spec["model_id"],
                        "backend": "hf",
                        "provider": "huggingface",
                        "status": "completed",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                run_status.append(
                    {
                        "model_key": model_key,
                        "model_id": spec["model_id"],
                        "backend": "hf",
                        "provider": "huggingface",
                        "status": "load_or_run_failed",
                        "error": str(exc),
                    }
                )
            finally:
                if adapter is not None:
                    del adapter
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    task_eval = {
        "zero_shot": {
            "task": "anomaly",
            "qa": "samples/anomaly_zero_shot_20.jsonl",
            "regime": "zero_shot",
        },
        "one_shot": {
            "task": "anomaly",
            "qa": "samples/anomaly_one_shot_20.jsonl",
            "regime": "one_shot",
        },
        "structured": {
            "task": "explainability",
            "qa": "samples/defect_20.jsonl",
            "regime": None,
        },
    }
    completed = [row for row in run_status if row["status"] == "completed"]
    for status in completed:
        model_key = status["model_key"]
        model_dir = run_root / model_key
        metrics_dir = model_dir / "metrics"
        for task_name, cfg in task_eval.items():
            cmd = [
                sys.executable,
                "scripts/evaluate.py",
                "--task",
                cfg["task"],
                "--qa",
                cfg["qa"],
                "--predictions",
                str(model_dir / f"{task_name}.jsonl"),
                "--out-dir",
                str(metrics_dir),
            ]
            if cfg["regime"]:
                cmd += ["--regime", cfg["regime"]]
            subprocess.run(cmd, check=True)

    leaderboard = []
    for status in completed:
        model_key = status["model_key"]
        metrics_dir = run_root / model_key / "metrics"
        anomaly = pd.read_csv(metrics_dir / "anomaly_overall.csv")
        explain = pd.read_csv(metrics_dir / "explainability_overall.csv").iloc[0]
        zero = anomaly.loc[anomaly["regime"] == "zero_shot"].iloc[0]
        one = anomaly.loc[anomaly["regime"] == "one_shot"].iloc[0]
        leaderboard.append(
            {
                "model_key": model_key,
                "model_id": status["model_id"],
                "backend": status["backend"],
                "provider": status["provider"],
                "zero_bacc": zero["balanced_accuracy"],
                "zero_f1": zero["f1"],
                "zero_invalid": zero["invalid_output_rate"],
                "one_bacc": one["balanced_accuracy"],
                "one_f1": one["f1"],
                "one_invalid": one["invalid_output_rate"],
                "defect_bacc": explain["defect_balanced_accuracy"],
                "defect_macro_f1": explain["defect_macro_f1"],
                "joint_accuracy": explain["joint_accuracy"],
                "false_defect_rate": explain["false_defect_rate"],
                "consistency_error": explain["consistency_error"],
                "structured_invalid": explain["invalid_output_rate"],
            }
        )
    leaderboard_df = pd.DataFrame(leaderboard)
    leaderboard_df.to_csv(run_root / "leaderboard.csv", index=False)
    display(leaderboard_df)

    review_rows = []
    for status in completed:
        model_key = status["model_key"]
        for task_name, gt_rows in qa_sets.items():
            pred_rows = read_jsonl(run_root / model_key / f"{task_name}.jsonl")
            pred_by_id = {row["id"]: row for row in pred_rows}
            for gt in gt_rows:
                pred = pred_by_id.get(gt["id"], {})
                review_rows.append(
                    {
                        "model_key": model_key,
                        "model_id": status["model_id"],
                        "backend": status["backend"],
                        "task": task_name,
                        "id": gt["id"],
                        "asset_category": gt.get("asset_category"),
                        "ground_truth": json.dumps(gt["answer"], ensure_ascii=False),
                        "output": pred.get("output", ""),
                        "error": pred.get("error"),
                        "latency_s": pred.get("latency_s"),
                    }
                )
    pd.DataFrame(review_rows).to_csv(run_root / "prediction_review.csv", index=False)

    metadata = {
        "benchmark_commit": benchmark_commit,
        "branch": "main",
        "smoke_limit_per_task": SMOKE_LIMIT,
        "qa_counts": {key: len(rows) for key, rows in qa_sets.items()},
        "api_models_requested": API_MODELS_TO_RUN,
        "hf_models_requested": HF_MODELS_TO_RUN,
        "api_key_presence": {"luxia": bool(luxia_api_key)},
        "luxia_external_chat_url": luxia_chat_url,
        "api_probe_status": api_probe_status,
        "gpu": gpu_name,
        "vram_gb": round(vram_gb, 2),
        "allow_hf_below_20gb": ALLOW_HF_BELOW_20GB,
        "model_registry": MODEL_REGISTRY,
        "source_zip_paths": [str(unsup_zip), str(sup_zip)],
        "result_root": str(run_root),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (run_root / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_root / "run_status.json").write_text(
        json.dumps(run_status, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    zip_path = Path(str(run_root) + ".zip")
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(run_root), "zip", root_dir=run_root)
    print("results directory:", run_root)
    print("result ZIP:", zip_path)
    print("completed models:", [row["model_key"] for row in completed])
    print("skipped/failed:", [row for row in run_status if row["status"] != "completed"])
    files.download(str(zip_path))


if __name__ == "__main__":
    main()
