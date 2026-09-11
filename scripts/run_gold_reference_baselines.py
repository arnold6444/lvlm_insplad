from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_gold_baselines as base  # noqa: E402


def prompt_for_reference(row: dict) -> str:
    options = row.get("options")
    if not isinstance(options, dict) or not options:
        raise ValueError(f"QA row {row.get('id')} has no options")
    option_lines = "\n".join(f"{k}. {v}" for k, v in options.items())
    return (
        "You are an industrial visual inspector. Image 1 is a verified NORMAL reference image "
        "of the same asset category. Image 2 is the TARGET image. Use Image 1 only as the normal "
        "visual standard and answer the question about Image 2. Do not judge Image 1. Do not use "
        "hidden metadata or external knowledge.\n\n"
        f"Question: {row['question']}\n"
        f"{option_lines}\n\n"
        "Return exactly one option letter and nothing else."
    )


def reference_manifest_rows(rows: list[dict]) -> list[dict]:
    by_core: dict[str, dict] = {}
    for row in rows:
        core_id = str(row.get("core_id") or "")
        asset = str(row.get("asset_category") or "")
        target = str(row.get("image") or "")
        unsup_target = str(row.get("unsupervised_image") or "")
        reference = str(row.get("normal_reference_image") or "")
        if not all([core_id, asset, target, unsup_target, reference]):
            raise RuntimeError(f"{row.get('id')}: missing target/reference metadata")
        if reference == unsup_target:
            raise RuntimeError(f"{row.get('id')}: reference image equals target image")
        current = {
            "core_id": core_id,
            "asset_category": asset,
            "target_image": target,
            "unsupervised_target_image": unsup_target,
            "normal_reference_image": reference,
        }
        previous = by_core.get(core_id)
        if previous is not None and previous != current:
            raise RuntimeError(f"{core_id}: inconsistent reference metadata across Q1/Q2/Q3")
        by_core[core_id] = current
    return [by_core[key] for key in sorted(by_core)]


def reference_summary_rows(manifest_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in manifest_rows:
        grouped[str(row["asset_category"])].append(row)
    out: list[dict] = []
    for asset in sorted(grouped):
        rows = grouped[asset]
        refs = sorted({str(row["normal_reference_image"]) for row in rows})
        out.append(
            {
                "asset_category": asset,
                "target_core_count": len(rows),
                "unique_reference_count": len(refs),
                "reference_images": "|".join(refs),
            }
        )
    return out


def reference_identity(manifest_rows: list[dict]) -> str:
    payload = json.dumps(
        manifest_rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepare_images(
    rows: list[dict],
    sup_zip: Path,
    unsup_zip: Path,
    image_root: Path,
) -> tuple[dict[str, Path], dict[str, Path]]:
    target_paths: dict[str, Path] = {}
    reference_paths: dict[str, Path] = {}
    for row in rows:
        target_member = str(row["image"])
        reference_member = str(row["normal_reference_image"])
        if target_member not in target_paths:
            target_paths[target_member] = base.safe_extract(
                sup_zip, target_member, image_root / "target"
            )
        if reference_member not in reference_paths:
            reference_paths[reference_member] = base.safe_extract(
                unsup_zip, reference_member, image_root / "reference"
            )
    return target_paths, reference_paths


def infer_luxia_reference(
    reference_path: Path,
    target_path: Path,
    prompt: str,
    api_key: str,
) -> tuple[str, float]:
    from insplad_benchmark.luxia_adapter import infer_luxia_pipeline

    started = time.time()
    result = infer_luxia_pipeline(
        [reference_path, target_path],
        prompt,
        api_key,
    )
    return str(result["output"]), round(time.time() - started, 3)


def run_model(
    model_key: str,
    rows: list[dict],
    target_paths: dict[str, Path],
    reference_paths: dict[str, Path],
    out_dir: Path,
    checkpoint_every: int,
) -> dict:
    spec = base.MODEL_REGISTRY[model_key]
    pred_path = out_dir / "predictions" / f"{model_key}.jsonl"
    existing_rows = base.read_jsonl(pred_path) if pred_path.is_file() else []
    existing = {str(r["id"]): r for r in existing_rows if r.get("id")}
    predictions: list[dict] = []

    adapter = None
    if spec["backend"] == "hf":
        from insplad_benchmark.hf_adapters import load_hf_adapter

        adapter = load_hf_adapter(spec)

    api_key = os.environ.get("LUXIA_API_KEY", "")
    if spec["backend"] == "luxia" and not api_key:
        raise RuntimeError("LUXIA_API_KEY is required for Luxia")

    called = reused = errors = 0
    total_latency = 0.0

    try:
        for index, row in enumerate(rows, 1):
            old = existing.get(str(row["id"]))
            if base.reusable_prediction(old):
                predictions.append(old)
                reused += 1
                continue

            prompt = prompt_for_reference(row)
            target_path = target_paths[str(row["image"])]
            reference_path = reference_paths[str(row["normal_reference_image"])]
            started = time.time()
            error = None
            output = ""
            try:
                if spec["backend"] == "hf":
                    from insplad_benchmark.hf_adapters import infer_hf

                    output = infer_hf(adapter, [reference_path, target_path], prompt)
                    latency = round(time.time() - started, 3)
                else:
                    output, latency = infer_luxia_reference(
                        reference_path, target_path, prompt, api_key
                    )
            except Exception as exc:
                latency = round(time.time() - started, 3)
                error = f"{type(exc).__name__}: {exc}"
                errors += 1

            valid_letters = {str(k).upper() for k in row["options"]}
            choice = base.parse_choice(output, valid_letters)
            predictions.append(
                {
                    "id": row["id"],
                    "core_id": row["core_id"],
                    "asset_category": row["asset_category"],
                    "question_type": row["question_type"],
                    "output": output,
                    "predicted_choice": choice,
                    "latency_s": latency,
                    "error": error,
                    "condition": "fixed_normal_reference",
                    "normal_reference_image": row["normal_reference_image"],
                }
            )
            called += 1
            total_latency += latency

            if checkpoint_every > 0 and called % checkpoint_every == 0:
                base.write_jsonl(pred_path, predictions)
                print(f"[{model_key}] {index}/{len(rows)} checkpoint")

        base.write_jsonl(pred_path, predictions)
    finally:
        if adapter is not None:
            del adapter
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    return {
        "model_key": model_key,
        "called_this_run": called,
        "reused_predictions": reused,
        "row_errors": errors,
        "called_latency_s": round(total_latency, 3),
    }


def evaluate_model(
    model_key: str,
    rows: list[dict],
    pred_path: Path,
) -> tuple[dict, list[dict]]:
    metric, by_asset = base.evaluate_model(model_key, rows, pred_path)
    q3_counts = Counter(
        str(row["asset_category"])
        for row in rows
        if row["question_type"] == "defect_localization"
    )
    for item in by_asset:
        count = q3_counts.get(str(item["asset_category"]), 0)
        item["q3_questions"] = count
        if count == 0:
            item["q3_accuracy"] = None
    return metric, by_asset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-dir", type=Path, required=True)
    parser.add_argument("--sup-zip", type=Path, required=True)
    parser.add_argument("--unsup-zip", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llava_onevision_7b", "minicpm_v_4_5", "qwen3_5_4b"],
        choices=list(base.MODEL_REGISTRY),
    )
    parser.add_argument("--checkpoint-every", type=int, default=5)
    args = parser.parse_args()

    rows = base.load_gold_qa(args.qa_dir)
    manifest_rows = reference_manifest_rows(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    snapshot = args.out_dir / "run_qa.jsonl"
    if snapshot.is_file():
        existing = base.read_jsonl(snapshot)
        if base.qa_identity(existing) != base.qa_identity(rows):
            raise RuntimeError("out-dir contains a different frozen QA snapshot")
    else:
        base.write_jsonl(snapshot, rows)

    manifest_path = args.out_dir / "reference_manifest.csv"
    base.write_csv(manifest_path, manifest_rows)
    base.write_csv(
        args.out_dir / "reference_summary.csv",
        reference_summary_rows(manifest_rows),
    )

    target_paths, reference_paths = prepare_images(
        rows,
        args.sup_zip,
        args.unsup_zip,
        args.out_dir / "_images",
    )

    run_status: list[dict] = []
    leaderboard: list[dict] = []
    by_asset_all: list[dict] = []

    for model_key in args.models:
        print(f"\n=== {model_key} / fixed normal reference ===")
        try:
            status = run_model(
                model_key,
                rows,
                target_paths,
                reference_paths,
                args.out_dir,
                args.checkpoint_every,
            )
            metric, by_asset = evaluate_model(
                model_key,
                rows,
                args.out_dir / "predictions" / f"{model_key}.jsonl",
            )
            metric["condition"] = "fixed_normal_reference"
            status["status"] = "completed"
            leaderboard.append(metric)
            by_asset_all.extend(by_asset)
        except Exception as exc:
            status = {
                "model_key": model_key,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(status["error"])
        run_status.append(status)
        base.write_csv(args.out_dir / "leaderboard.csv", leaderboard)
        base.write_csv(args.out_dir / "metrics_by_asset.csv", by_asset_all)
        (args.out_dir / "run_status.json").write_text(
            json.dumps(run_status, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    metadata = {
        "qa_dir": str(args.qa_dir),
        "qa_count": len(rows),
        "qa_identity_sha256": base.qa_identity(rows),
        "reference_identity_sha256": reference_identity(manifest_rows),
        "reference_core_count": len(manifest_rows),
        "models_requested": args.models,
        "model_registry": base.MODEL_REGISTRY,
        "condition": "fixed_normal_reference_final_gold_v1",
        "image_order": ["verified_normal_reference", "target"],
        "q3_scope_note": "Q3 is evaluated only on the available localization subset.",
    }
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
