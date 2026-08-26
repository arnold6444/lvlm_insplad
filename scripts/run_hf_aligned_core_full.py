from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from insplad_benchmark.hf_adapters import infer_hf, load_hf_adapter
from insplad_benchmark.sampling import balanced_binary_sample

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

LEGACY_EXPERIMENT = "legacy"
ONE_SHOT_PLUS_EXPERIMENT = "one_shot_plus"
ONE_SHOT_PLUS_MODELS = (
    "qwen3_5_4b",
    "internvl3_5_8b",
    "minicpm_v_4_5",
)
DEFAULT_ONE_SHOT_PLUS_SMOKE_SIZE = 25
ANOMALY_TASKS = {"zero_shot", "one_shot", "one_shot_plus"}
RUN_CONTRACT_SCHEMA_VERSION = 1
RUN_CONTRACT_FILENAME = "run_contract.json"
LEGACY_OUTPUT_FILENAMES = {
    "zero_shot.jsonl",
    "one_shot.jsonl",
    "structured.jsonl",
    "explainability_overall.csv",
    "explainability_by_asset.csv",
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


def resolve_models(experiment: str, requested_models: list[str] | None) -> list[str]:
    models = list(requested_models) if requested_models is not None else (
        list(ONE_SHOT_PLUS_MODELS)
        if experiment == ONE_SHOT_PLUS_EXPERIMENT
        else list(MODEL_REGISTRY)
    )
    unknown = sorted(set(models) - set(MODEL_REGISTRY))
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    if experiment == ONE_SHOT_PLUS_EXPERIMENT:
        unsupported = sorted(set(models) - set(ONE_SHOT_PLUS_MODELS))
        if unsupported:
            raise ValueError(
                "one_shot_plus supports only Qwen/InternVL/MiniCPM; "
                f"unsupported models: {unsupported}"
            )
    return models


def select_one_shot_plus_rows(
    rows: list[dict],
    *,
    full: bool,
    smoke_size: int = DEFAULT_ONE_SHOT_PLUS_SMOKE_SIZE,
) -> list[dict]:
    if len(rows) != 1378:
        raise RuntimeError(f"Aligned Core one_shot_plus QA must be 1378, got {len(rows)}")

    ids = [str(row.get("id", "")) for row in rows]
    if any(not row_id for row_id in ids) or len(set(ids)) != len(ids):
        raise RuntimeError("one_shot_plus QA IDs must be present and unique")
    for row in rows:
        missing = [
            field
            for field in ("image", "reference_image", "retrieval_method")
            if not row.get(field)
        ]
        if missing:
            raise RuntimeError(
                f"one_shot_plus QA row {row['id']} missing required fields: {missing}"
            )

    if full:
        return [dict(row) for row in rows]
    if not 20 <= smoke_size <= 50:
        raise ValueError("one_shot_plus smoke size must be between 20 and 50")
    return balanced_binary_sample(rows, smoke_size, seed=42)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def qa_identity_sha256(rows: list[dict]) -> str:
    """Return an order-sensitive identity for the exact selected QA payload."""
    payload = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def one_shot_plus_run_contract(
    rows: list[dict],
    *,
    full: bool,
    smoke_size: int,
) -> dict:
    return {
        "schema_version": RUN_CONTRACT_SCHEMA_VERSION,
        "experiment": ONE_SHOT_PLUS_EXPERIMENT,
        "run_mode": "full" if full else "smoke",
        "smoke_size": None if full else smoke_size,
        "selected_qa_count": len(rows),
        "selected_qa_identity_sha256": qa_identity_sha256(rows),
    }


def _load_json_object(path: Path, *, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid {label} at {path}: expected a JSON object")
    return value


def _assert_contract_matches(actual: dict, expected: dict, *, source: Path) -> None:
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "one_shot_plus output directory has an incompatible run contract "
            f"at {source}: {mismatches}"
        )


def _assert_metadata_matches(metadata: dict, expected: dict, *, source: Path) -> None:
    required = {
        "experiment": expected["experiment"],
        "run_mode": expected["run_mode"],
        "selected_qa_count": expected["selected_qa_count"],
    }
    if "smoke_size" in metadata:
        required["smoke_size"] = expected["smoke_size"]
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in required.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "one_shot_plus output directory has incompatible run metadata "
            f"at {source}: {mismatches}"
        )


def _assert_snapshot_matches(snapshot_path: Path, expected: dict) -> None:
    if not snapshot_path.is_file():
        raise RuntimeError(
            "one_shot_plus output directory is missing its selected QA snapshot: "
            f"{snapshot_path}"
        )
    try:
        snapshot_rows = read_jsonl(snapshot_path)
        identity = qa_identity_sha256(snapshot_rows)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"invalid one_shot_plus selected QA snapshot at {snapshot_path}: {exc}"
        ) from exc
    if (
        len(snapshot_rows) != expected["selected_qa_count"]
        or identity != expected["selected_qa_identity_sha256"]
    ):
        raise RuntimeError(
            "one_shot_plus output directory has a different selected QA identity "
            f"at {snapshot_path}"
        )


def _legacy_output_artifacts(out_dir: Path) -> list[Path]:
    if not out_dir.is_dir():
        return []
    return sorted(
        (
            path
            for path in out_dir.rglob("*")
            if path.is_file() and path.name in LEGACY_OUTPUT_FILENAMES
        ),
        key=lambda path: path.as_posix(),
    )


def ensure_one_shot_plus_output_contract(
    out_dir: Path,
    rows: list[dict],
    *,
    full: bool,
    smoke_size: int,
) -> dict:
    """Claim or validate a result root before any mutable runner output."""
    expected = one_shot_plus_run_contract(
        rows,
        full=full,
        smoke_size=smoke_size,
    )
    contract_path = out_dir / RUN_CONTRACT_FILENAME
    metadata_path = out_dir / "run_metadata.json"
    snapshot_path = out_dir / "run_qa" / "one_shot_plus.jsonl"

    if out_dir.exists() and not out_dir.is_dir():
        raise RuntimeError(f"one_shot_plus out-dir is not a directory: {out_dir}")

    legacy_artifacts = _legacy_output_artifacts(out_dir)
    if legacy_artifacts:
        preview = [str(path) for path in legacy_artifacts[:5]]
        raise RuntimeError(
            "refusing to mix one_shot_plus with legacy output artifacts: "
            f"{preview}"
        )

    if contract_path.is_file():
        _assert_contract_matches(
            _load_json_object(contract_path, label="run contract"),
            expected,
            source=contract_path,
        )
        if metadata_path.exists():
            if not metadata_path.is_file():
                raise RuntimeError(f"invalid run metadata path: {metadata_path}")
            _assert_metadata_matches(
                _load_json_object(metadata_path, label="run metadata"),
                expected,
                source=metadata_path,
            )
        if snapshot_path.exists():
            _assert_snapshot_matches(snapshot_path, expected)
        return expected
    if contract_path.exists():
        raise RuntimeError(f"invalid run contract path: {contract_path}")

    existing_entries = list(out_dir.iterdir()) if out_dir.is_dir() else []
    if existing_entries:
        # Migrate only a prior one_shot_plus result root whose metadata and
        # immutable selected-QA snapshot prove the same run identity.
        if not metadata_path.is_file() or not snapshot_path.is_file():
            raise RuntimeError(
                "refusing nonempty one_shot_plus out-dir without a compatible "
                f"run contract or metadata/snapshot pair: {out_dir}"
            )
        _assert_metadata_matches(
            _load_json_object(metadata_path, label="run metadata"),
            expected,
            source=metadata_path,
        )
        _assert_snapshot_matches(snapshot_path, expected)

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with contract_path.open("x", encoding="utf-8") as handle:
            json.dump(expected, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except FileExistsError:
        # A concurrent invocation may have claimed the empty directory. It is
        # safe only if it claimed the exact same immutable run identity.
        _assert_contract_matches(
            _load_json_object(contract_path, label="run contract"),
            expected,
            source=contract_path,
        )
    return expected


def ensure_one_shot_plus_snapshot(snapshot_path: Path, rows: list[dict]) -> None:
    expected = {
        "selected_qa_count": len(rows),
        "selected_qa_identity_sha256": qa_identity_sha256(rows),
    }
    if snapshot_path.exists():
        _assert_snapshot_matches(snapshot_path, expected)
        return
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(snapshot_path, rows)


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


def reusable_prediction(
    row: dict | None,
    *,
    reference_image: str | None = None,
    retrieval_method: str | None = None,
) -> bool:
    if not (row and row.get("output")) or row.get("error"):
        return False
    if reference_image is not None and row.get("reference_image") != reference_image:
        return False
    if retrieval_method is not None and row.get("retrieval_method") != retrieval_method:
        return False
    return True


def prediction_is_reusable(task: str, row: dict | None, gt: dict) -> bool:
    if task != ONE_SHOT_PLUS_EXPERIMENT:
        return reusable_prediction(row)
    return reusable_prediction(
        row,
        reference_image=str(gt["reference_image"]),
        retrieval_method=str(gt["retrieval_method"]),
    )


def task_inputs(task: str, row: dict, path_cache: dict[str, Path]) -> tuple[list[Path], str]:
    if task == "zero_shot":
        return [path_cache[row["image"]]], ANOMALY_PROMPT
    if task in {"one_shot", ONE_SHOT_PLUS_EXPERIMENT}:
        return [path_cache[row["reference_image"]], path_cache[row["image"]]], ONE_SHOT_PROMPT
    return [path_cache[row["image"]]], structured_prompt(row)


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
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=[LEGACY_EXPERIMENT, ONE_SHOT_PLUS_EXPERIMENT],
        default=LEGACY_EXPERIMENT,
    )
    parser.add_argument("--qa-dir", type=Path, required=True)
    parser.add_argument("--unsup-zip", type=Path, required=True)
    parser.add_argument("--sup-zip", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("/content/insplad_aligned_core_images"))
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    run_size = parser.add_mutually_exclusive_group()
    run_size.add_argument("--full", action="store_true")
    run_size.add_argument("--smoke-size", type=int)
    args = parser.parse_args()

    if args.experiment == LEGACY_EXPERIMENT and (args.full or args.smoke_size is not None):
        parser.error("--full/--smoke-size are available only for --experiment one_shot_plus")
    try:
        args.models = resolve_models(args.experiment, args.models)
    except ValueError as exc:
        parser.error(str(exc))

    qa_paths: dict[str, Path]
    if args.experiment == LEGACY_EXPERIMENT:
        qa_paths = {
            "zero_shot": args.qa_dir / "zero_shot.jsonl",
            "one_shot": args.qa_dir / "one_shot.jsonl",
            "structured": args.qa_dir / "structured.jsonl",
        }
        qa = {task: read_jsonl(path) for task, path in qa_paths.items()}
        if not all(len(rows) == 1378 for rows in qa.values()):
            raise RuntimeError(f"Aligned Core QA must be 1378/1378/1378, got { {k: len(v) for k, v in qa.items()} }")
        if [r["id"] for r in qa["zero_shot"]] != [r["id"] for r in qa["one_shot"]]:
            raise RuntimeError("zero-shot and one-shot query IDs differ")
    else:
        full_rows = read_jsonl(args.qa_dir / "one_shot_plus.jsonl")
        smoke_size = (
            args.smoke_size
            if args.smoke_size is not None
            else DEFAULT_ONE_SHOT_PLUS_SMOKE_SIZE
        )
        try:
            selected_rows = select_one_shot_plus_rows(
                full_rows,
                full=args.full,
                smoke_size=smoke_size,
            )
        except ValueError as exc:
            parser.error(str(exc))
        qa = {ONE_SHOT_PLUS_EXPERIMENT: selected_rows}
        qa_paths = {}

    if args.experiment == ONE_SHOT_PLUS_EXPERIMENT:
        ensure_one_shot_plus_output_contract(
            args.out_dir,
            qa[ONE_SHOT_PLUS_EXPERIMENT],
            full=args.full,
            smoke_size=smoke_size,
        )
        snapshot_path = args.out_dir / "run_qa" / "one_shot_plus.jsonl"
        ensure_one_shot_plus_snapshot(
            snapshot_path,
            qa[ONE_SHOT_PLUS_EXPERIMENT],
        )
        qa_paths[ONE_SHOT_PLUS_EXPERIMENT] = snapshot_path
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    args.image_root.mkdir(parents=True, exist_ok=True)

    path_cache: dict[str, Path] = {}
    for task, rows in qa.items():
        for row in rows:
            if task in ANOMALY_TASKS:
                if row["image"] not in path_cache:
                    path_cache[row["image"]] = safe_extract_member(args.unsup_zip, row["image"], args.image_root, "unsup")
                ref = row.get("reference_image")
                if ref and ref not in path_cache:
                    path_cache[ref] = safe_extract_member(args.unsup_zip, ref, args.image_root, "unsup")
            elif row["image"] not in path_cache:
                path_cache[row["image"]] = safe_extract_member(args.sup_zip, row["image"], args.image_root, "sup")

    import torch
    from tqdm.auto import tqdm
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
                    if prediction_is_reusable(task, old, gt):
                        predictions.append(old)
                        reused += 1
                    else:
                        image_paths, prompt = task_inputs(task, gt, path_cache)
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
                        prediction = {
                            "id": gt["id"], "output": output, "error": error,
                            "latency_s": round(time.time() - started, 3),
                            "model_key": model_key, "model_id": spec["model_id"],
                            "backend": "hf", "provider": "huggingface",
                        }
                        if task == ONE_SHOT_PLUS_EXPERIMENT:
                            prediction.update({
                                "reference_image": gt["reference_image"],
                                "retrieval_method": gt["retrieval_method"],
                            })
                        predictions.append(prediction)
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

    if args.experiment == LEGACY_EXPERIMENT:
        task_eval = {
            "zero_shot": ("anomaly", args.qa_dir / "zero_shot.jsonl", "zero_shot"),
            "one_shot": ("anomaly", args.qa_dir / "one_shot.jsonl", "one_shot"),
            "structured": ("explainability", args.qa_dir / "structured.jsonl", None),
        }
        scoring_models = MODEL_REGISTRY.items()
    else:
        task_eval = {
            ONE_SHOT_PLUS_EXPERIMENT: (
                "anomaly",
                qa_paths[ONE_SHOT_PLUS_EXPERIMENT],
                ONE_SHOT_PLUS_EXPERIMENT,
            ),
        }
        # Include any allowed model already completed in this result root, even
        # when the current invocation requested only one model. This preserves
        # a cumulative leaderboard across resumable Colab sessions.
        scoring_models = (
            (key, MODEL_REGISTRY[key]) for key in ONE_SHOT_PLUS_MODELS
        )
    leaderboard = []
    for model_key, spec in scoring_models:
        model_dir = args.out_dir / model_key
        ready = True
        for task, gt_rows in qa.items():
            pred_path = model_dir / f"{task}.jsonl"
            if not pred_path.is_file():
                ready = False
                break
            by_id = {r["id"]: r for r in read_jsonl(pred_path)}
            if len(by_id) != len(gt_rows) or not all(
                prediction_is_reusable(task, by_id.get(gt["id"]), gt)
                for gt in gt_rows
            ):
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
        if args.experiment == ONE_SHOT_PLUS_EXPERIMENT:
            plus = anomaly.loc[anomaly["regime"] == ONE_SHOT_PLUS_EXPERIMENT].iloc[0]
            asset_macro = pd.read_csv(metrics_dir / "anomaly_asset_macro.csv")
            plus_macro = asset_macro.loc[
                asset_macro["regime"] == ONE_SHOT_PLUS_EXPERIMENT
            ].iloc[0]
            leaderboard.append({
                "model_key": model_key,
                "model_id": spec["model_id"],
                "one_shot_plus_bacc": plus["balanced_accuracy"],
                "one_shot_plus_f1": plus["f1"],
                "one_shot_plus_asset_macro_bacc": plus_macro["balanced_accuracy"],
                "one_shot_plus_asset_macro_f1": plus_macro["f1"],
                "one_shot_plus_invalid": plus["invalid_output_rate"],
            })
        else:
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
    if args.experiment == ONE_SHOT_PLUS_EXPERIMENT:
        metadata.update({
            "experiment": ONE_SHOT_PLUS_EXPERIMENT,
            "run_mode": "full" if args.full else "smoke",
            "smoke_size": None if args.full else smoke_size,
            "selected_qa_count": len(qa[ONE_SHOT_PLUS_EXPERIMENT]),
            "selected_qa_snapshot": str(qa_paths[ONE_SHOT_PLUS_EXPERIMENT]),
            "models": {key: MODEL_REGISTRY[key] for key in args.models},
            "allowed_models": {
                key: MODEL_REGISTRY[key] for key in ONE_SHOT_PLUS_MODELS
            },
        })
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(leaderboard))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
