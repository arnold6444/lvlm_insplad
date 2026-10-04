#!/usr/bin/env python3
"""Gold-v2 Normal Reference + Vision Expert evidence (next stage after vision expert full).

Every LVLM sees ``[normal reference, target, (expert ROI/heatmap)]`` plus an optional
score cue, for the selected ``expert:condition`` pairs. Results are compared against
image-only, normal-reference-only and expert-only rows already produced on the same
246-image gold-v2 set.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_gold_baselines as base  # noqa: E402
import run_gold_v2_baselines as v2  # noqa: E402
import run_gold_v2_vision_expert_full as vx  # noqa: E402
from insplad_benchmark.vision_expert import read_manifest, resolve_artifact_path  # noqa: E402

REGIME = "normal_reference+expert"
METRICS = ("overall_accuracy", "anomaly_f1", "defect_macro_f1", "localization_accuracy")


def parse_pairs(values: Sequence[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for value in values:
        expert, sep, condition = str(value).partition(":")
        if not sep or expert not in vx.EXPERTS or condition not in vx.CONDITIONS:
            raise ValueError(f"pair must be <expert>:<condition> from known values, got {value!r}")
        if (expert, condition) not in pairs:
            pairs.append((expert, condition))
    if not pairs:
        raise ValueError("at least one expert:condition pair is required")
    return pairs


def build_prompt(row: Mapping[str, object], condition: str, expert: str, percentile: float) -> str:
    options = "\n".join(f"{k}. {v}" for k, v in dict(row["options"]).items())
    images = (
        "Image 1 is a verified NORMAL reference of the same asset category. "
        "Image 2 is the complete TARGET image. "
    )
    score_text = (
        f"The {expert} vision-expert anomaly score of the TARGET is at the "
        f"{percentile:.1f}th percentile among same-asset images in this fixed "
        "unlabeled benchmark cohort. Higher is more suspicious. "
        "This relative score uses no ground-truth label and may be wrong. "
    )
    roi_text = (
        f"Image 3 is a candidate ROI crop of the TARGET selected automatically by {expert}. "
        "It may highlight a normal region. "
    )
    heatmap_text = (
        f"Image 3 is the {expert} anomaly heatmap overlay of the TARGET; "
        "warmer regions are more suspicious. It may be noisy. "
    )
    evidence = {
        "score": score_text,
        "roi": roi_text,
        "heatmap": heatmap_text,
        "score_roi": score_text + roi_text,
        "score_heatmap": score_text + heatmap_text,
    }[condition]
    return (
        "You are an industrial visual inspector. "
        + images
        + "Use Image 1 only as the normal visual standard. "
        + evidence
        + "Answer the multiple-choice question about the TARGET. "
        "Do not assume the vision expert is correct.\n\n"
        f"Question: {row['question']}\n{options}\n\n"
        "Return exactly one option letter and nothing else."
    )


def build_images(
    condition: str,
    normal: Path,
    target: Path,
    expert_row: Mapping[str, str],
    manifest_path: Path,
    artifact_root: Path,
) -> list[Path]:
    # vx.evidence_images returns [target] or [target, evidence]; the normal reference goes first.
    return [normal, *vx.evidence_images(condition, target, expert_row, manifest_path, artifact_root)]


def normal_reference_by_core(rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    refs: dict[str, str] = {}
    for row in rows:
        core_id = str(row["core_id"])
        value = str(row.get("normal_reference_image") or "")
        if not value:
            continue
        if refs.get(core_id, value) != value:
            raise RuntimeError(f"{core_id}: inconsistent normal_reference_image")
        refs[core_id] = value
    missing = sorted({str(row["core_id"]) for row in rows} - set(refs))
    if missing:
        raise RuntimeError(f"normal_reference_image missing for {len(missing)} cores, e.g. {missing[:3]}")
    return refs


def run_pair(
    model_key: str,
    expert: str,
    condition: str,
    rows: Sequence[dict],
    expert_index: Mapping[tuple[str, str], Mapping[str, str]],
    target_paths: Mapping[str, Path],
    normal_paths: Mapping[str, Path],
    normal_by_core: Mapping[str, str],
    manifest_path: Path,
    artifact_root: Path,
    out_dir: Path,
    checkpoint_every: int,
    holder: "base.LazyHFAdapter | None",
    luxia_key: str = "",
) -> tuple[Path, dict]:
    from insplad_benchmark.hf_adapters import infer_hf

    spec = base.MODEL_REGISTRY[model_key]
    pred_path = out_dir / "predictions" / model_key / f"{expert}__{condition}.jsonl"
    existing = {str(row["id"]): row for row in vx.read_jsonl(pred_path)}
    predictions: list[dict] = []
    called = reused = errors = 0
    for index, row in enumerate(rows, start=1):
        old = existing.get(str(row["id"]))
        if vx.prediction_reusable(old, model_key=model_key, expert=expert, condition=condition):
            predictions.append(old)
            reused += 1
            continue

        core_id = str(row["core_id"])
        erow = expert_index[(core_id, expert)]
        images = build_images(
            condition,
            normal_paths[normal_by_core[core_id]],
            target_paths[str(row["image"])],
            erow,
            manifest_path,
            artifact_root,
        )
        percentile = float(erow["score_percentile_full"])
        prompt_text = build_prompt(row, condition, expert, percentile)
        # Load outside the per-row try so a load failure stops the model, not every row.
        adapter = holder.get() if holder is not None else None
        started = time.time()
        output = ""
        error = ""
        try:
            if spec["backend"] == "hf":
                output = str(infer_hf(adapter, images, prompt_text))
                latency = round(time.time() - started, 3)
            else:
                output, latency = vx.infer_luxia(images, prompt_text, luxia_key)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
            latency = round(time.time() - started, 3)
            errors += 1

        predictions.append(
            {
                "id": row["id"],
                "core_id": row["core_id"],
                "asset_category": row["asset_category"],
                "question_type": row["question_type"],
                "model_key": model_key,
                "model_id": spec["model_id"],
                "expert": expert,
                "condition": condition,
                "regime": REGIME,
                "score_percentile_full": percentile,
                "output": output,
                "predicted_choice": base.parse_choice(
                    output, {str(k).upper() for k in row["options"]}
                ),
                "latency_s": latency,
                "error": error,
            }
        )
        called += 1
        if checkpoint_every > 0 and called % checkpoint_every == 0:
            vx.write_jsonl(pred_path, predictions)
            print(f"[{model_key}/{expert}/{condition}] {index}/{len(rows)} checkpoint", flush=True)
    vx.write_jsonl(pred_path, predictions)
    status = {
        "model_key": model_key,
        "expert": expert,
        "condition": condition,
        "status": "complete",
        "called_this_run": called,
        "reused_predictions": reused,
        "row_errors": errors,
    }
    return pred_path, status


def build_comparison(
    leaderboard: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, str]],
    reference_rows: Sequence[Mapping[str, str]],
    expert_rows: Sequence[Mapping[str, str]],
) -> list[dict]:
    baseline = {str(r["model_key"]): r for r in baseline_rows}
    reference = {
        str(r["model_key"]): r for r in reference_rows if r.get("condition") == "normal_reference"
    }
    expert_only = {
        (str(r["model_key"]), str(r["expert"]), str(r["condition"])): r for r in expert_rows
    }
    out: list[dict] = []
    for row in leaderboard:
        model_key, expert, condition = str(row["model_key"]), str(row["expert"]), str(row["condition"])
        controls = {
            "image_only": baseline.get(model_key),
            "normal_reference": reference.get(model_key),
            "expert_only": expert_only.get((model_key, expert, condition)),
        }
        item: dict[str, object] = {"model_key": model_key, "expert": expert, "condition": condition}
        for metric in METRICS:
            try:
                combined = float(row[metric])
            except (KeyError, TypeError, ValueError):
                continue
            item[f"{metric}_combined"] = combined
            for name, control in controls.items():
                try:
                    value = float(control[metric]) if control else None
                except (KeyError, TypeError, ValueError):
                    value = None
                item[f"{metric}_{name}"] = value
                item[f"{metric}_delta_vs_{name}"] = (
                    round(combined - value, 6) if value is not None else None
                )
        out.append(item)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qa-dir", type=Path, required=True)
    parser.add_argument("--sup-zip", type=Path, required=True)
    parser.add_argument("--unsup-zip", type=Path, required=True)
    parser.add_argument("--roi-manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pairs", nargs="+", required=True, help="expert:condition, e.g. anomalyclip:score")
    hf_models = [key for key in vx.MODELS if base.MODEL_REGISTRY[key]["backend"] == "hf"]
    parser.add_argument("--models", nargs="+", default=hf_models, choices=list(vx.MODELS))
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--baseline-leaderboard", type=Path)
    parser.add_argument("--reference-leaderboard", type=Path)
    parser.add_argument("--expert-leaderboard", type=Path)
    parser.add_argument("--selection-note", default="")
    args = parser.parse_args()

    pairs = parse_pairs(args.pairs)
    experts = sorted({expert for expert, _ in pairs})
    rows = v2.load_gold_qa_v2(args.qa_dir)
    core_ids = {str(row["core_id"]) for row in rows}
    if len(core_ids) != 246:
        raise RuntimeError(f"expected 246 unique gold-v2 images, got {len(core_ids)}")
    normal_by_core = normal_reference_by_core(rows)

    manifest_rows = [dict(row) for row in read_manifest(args.roi_manifest)]
    # Percentiles are computed per expert/asset exactly as in the vision expert full run.
    selected = [
        row for row in manifest_rows
        if str(row.get("expert", "")) in experts and str(row.get("query_id", "")) in core_ids
    ]
    if len(selected) != len(core_ids) * len(experts):
        raise RuntimeError(
            f"ROI manifest coverage mismatch: {len(selected)} != {len(core_ids) * len(experts)}"
        )
    vx.add_score_percentiles(selected)
    expert_index = {(str(r["query_id"]), str(r["expert"])): r for r in selected}
    for row in selected:
        for field in ("roi_path", "overlay_path"):
            path = resolve_artifact_path(args.roi_manifest, str(row[field]), args.artifact_root)
            if not path.is_file():
                raise FileNotFoundError(path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    target_paths = base.prepare_images(rows, args.sup_zip, args.out_dir / "_images" / "target")
    normal_paths: dict[str, Path] = {}
    for member in sorted(set(normal_by_core.values())):
        normal_paths[member] = base.safe_extract(args.unsup_zip, member, args.out_dir / "_images" / "normal")

    leaderboard_path = args.out_dir / "leaderboard.csv"
    by_asset_path = args.out_dir / "metrics_by_asset.csv"
    status_path = args.out_dir / "run_status.json"
    status_json: list[dict] = []
    if status_path.is_file():
        try:
            status_json = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            status_json = []
    luxia_key = os.environ.get("LUXIA_API_KEY", "")

    for model_key in args.models:
        spec = base.MODEL_REGISTRY[model_key]
        if spec["backend"] != "hf" and not luxia_key:
            raise RuntimeError(f"{model_key}: LUXIA_API_KEY is required")
        holder = base.LazyHFAdapter(spec) if spec["backend"] == "hf" else None
        try:
            for expert, condition in pairs:
                print(f"\n=== {model_key} / normal_reference + {expert} / {condition} ===", flush=True)
                pred_path, status = run_pair(
                    model_key, expert, condition, rows, expert_index, target_paths,
                    normal_paths, normal_by_core, args.roi_manifest, args.artifact_root,
                    args.out_dir, args.checkpoint_every, holder, luxia_key,
                )
                metric, assets = base.evaluate_model(model_key, rows, pred_path)
                extra = {"expert": expert, "condition": condition, "regime": REGIME}
                metric.update(extra)
                for asset in assets:
                    asset.update(extra)
                vx.upsert_rows(leaderboard_path, [metric], ("model_key", "expert", "condition"))
                vx.upsert_rows(
                    by_asset_path, assets, ("model_key", "expert", "condition", "asset_category")
                )
                key = (model_key, expert, condition)
                status_json = [
                    item for item in status_json
                    if (str(item.get("model_key")), str(item.get("expert")), str(item.get("condition"))) != key
                ]
                status_json.append(status)
                status_path.write_text(
                    json.dumps(status_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
        finally:
            if holder is not None:
                holder.release()

    wanted = {(m, e, c) for m in args.models for e, c in pairs}
    leaderboard = [
        row for row in vx.read_csv(leaderboard_path)
        if (row.get("model_key"), row.get("expert"), row.get("condition")) in wanted
    ]
    if len(leaderboard) != len(wanted):
        raise RuntimeError(f"leaderboard coverage mismatch: {len(leaderboard)} != {len(wanted)}")

    def optional_rows(path: Path | None) -> list[dict[str, str]]:
        return vx.read_csv(path) if path else []

    comparison = build_comparison(
        leaderboard,
        optional_rows(args.baseline_leaderboard),
        optional_rows(args.reference_leaderboard),
        optional_rows(args.expert_leaderboard),
    )
    vx.write_csv(args.out_dir / "comparison.csv", comparison)
    metadata = {
        "protocol": "gold_v2_reference_expert",
        "regime": REGIME,
        "gold_images": len(core_ids),
        "qa_questions": len(rows),
        "models": args.models,
        "pairs": [f"{e}:{c}" for e, c in pairs],
        "image_order": ["normal_reference", "target", "expert_evidence (roi/heatmap conditions only)"],
        "heatmap_evidence": "pure heatmap overlay without ROI box (merged manifest overlay_path)",
        "score_definition": (
            "within-expert, within-asset percentile over the fixed unlabeled 246-image cohort; "
            "no GT labels used"
        ),
        "selection_note": args.selection_note,
    }
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
