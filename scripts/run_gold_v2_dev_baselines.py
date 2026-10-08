#!/usr/bin/env python3
"""Baselines on the gold-v2 development set (``scripts/build_gold_v2_dev_set.py``).

Three conditions with the gold v2 prompt builders, so dev numbers line up with the gold runs:

- ``image_only``          : ``run_gold_baselines.prompt_for``, images ``[target]``
- ``normal_ref_fixed``    : ``prompt_for_condition(..., "normal_reference")``, ``[fixed normal, target]``
- ``normal_ref_retrieved``: same prompt, ``[DINOv2 top-1 normal, target]``

Reports Q1 anomaly F1, Q2 defect macro-F1, Q1-Q2 contradiction, joint accuracy and FDR overall and per asset,
plus image-level paired bootstrap CIs for fixed vs retrieved reference and image-only vs each reference.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_gold_baselines as base  # noqa: E402
import run_gold_v2_consistency_metrics as cons  # noqa: E402
import run_gold_v2_reference_ablation as ref  # noqa: E402

MODELS = ("internvl3_5_8b", "qwen3_5_4b")
CONDITIONS = ("image_only", "normal_ref_fixed", "normal_ref_retrieved")
COMPARISONS = (("image_only", "normal_ref_fixed"), ("image_only", "normal_ref_retrieved"),
               ("normal_ref_fixed", "normal_ref_retrieved"))
PROMPT_VERSION = 1


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def build_prompt(row: Mapping, condition: str) -> str:
    if condition == "image_only":
        return base.prompt_for(dict(row))
    if condition in ("normal_ref_fixed", "normal_ref_retrieved"):
        return ref.prompt_for_condition(dict(row), "normal_reference")
    raise ValueError(condition)


def images_for(condition: str, target: Path, fixed: Path | None, retrieved: Path | None) -> list[Path]:
    if condition == "image_only":
        return [target]
    if condition == "normal_ref_fixed":
        return [fixed, target]
    if condition == "normal_ref_retrieved":
        return [retrieved, target]
    raise ValueError(condition)


def reusable(old: Mapping | None, model_key: str, condition: str) -> bool:
    return bool(old and not old.get("error") and str(old.get("output", "")).strip()
                and old.get("model_key") == model_key and old.get("condition") == condition
                and old.get("prompt_version") == PROMPT_VERSION)


def main(argv: Sequence[str] | None = None, *, loader: Callable | None = None, infer: Callable | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dev-qa", type=Path, required=True, help="dev_master_qa_*.jsonl")
    p.add_argument("--sup-zip", type=Path, required=True)
    p.add_argument("--unsup-zip", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)

    if infer is None:
        from insplad_benchmark.hf_adapters import infer_hf as infer  # noqa: N813
    rows = read_jsonl(args.dev_qa)
    if not rows or any(r.get("protocol") != "gold_v2_dev" for r in rows):
        raise RuntimeError("expected a gold_v2_dev question file")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = base.prepare_images(rows, args.sup_zip, args.out_dir / "_images" / "target")
    ref_members = sorted({str(r["normal_reference_image"]) for r in rows} |
                         {str(r["retrieved_reference_image"]) for r in rows})
    refs = {m: base.safe_extract(args.unsup_zip, m, args.out_dir / "_images" / "normal") for m in ref_members}

    statuses: list[dict] = []
    preds: dict[tuple[str, str], dict[str, str | None]] = {}
    for model_key in args.models:
        holder = base.LazyHFAdapter(base.MODEL_REGISTRY[model_key], loader=loader)
        try:
            for condition in args.conditions:
                print(f"\n=== {model_key} / {condition} ===", flush=True)
                pred_path = args.out_dir / "predictions" / model_key / f"{condition}.jsonl"
                existing = {str(r["id"]): r for r in (read_jsonl(pred_path) if pred_path.is_file() else [])}
                out_rows: list[dict] = []
                called = reused = errors = 0
                for row in rows:
                    old = existing.get(str(row["id"]))
                    if reusable(old, model_key, condition):
                        out_rows.append(old)
                        reused += 1
                        continue
                    images = images_for(condition, targets[str(row["image"])], refs[str(row["normal_reference_image"])],
                                        refs[str(row["retrieved_reference_image"])])
                    adapter = holder.get()
                    started = time.time()
                    output, error = "", ""
                    try:
                        output = str(infer(adapter, images, build_prompt(row, condition)))
                    except Exception as exc:  # retried on the next run
                        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
                        errors += 1
                    out_rows.append({
                        "id": row["id"], "core_id": row["core_id"], "model_key": model_key, "condition": condition,
                        "prompt_version": PROMPT_VERSION, "output": output,
                        "predicted_choice": base.parse_choice(output, {str(k).upper() for k in row["options"]}),
                        "latency_s": round(time.time() - started, 3), "error": error,
                    })
                    called += 1
                    if args.checkpoint_every > 0 and called % args.checkpoint_every == 0:
                        base.write_jsonl(pred_path, out_rows)
                        print(f"[{model_key}/{condition}] {len(out_rows)}/{len(rows)} checkpoint", flush=True)
                base.write_jsonl(pred_path, out_rows)
                preds[(model_key, condition)] = {str(r["id"]): r.get("predicted_choice") for r in out_rows}
                statuses.append({"model_key": model_key, "condition": condition, "called_this_run": called,
                                 "reused_predictions": reused, "row_errors": errors,
                                 "status": "complete" if errors == 0 else "has_errors"})
                if errors:
                    first = next(r["error"] for r in out_rows if r.get("error"))
                    print(f"[{model_key}/{condition}] {errors} errors, first: {first[:300]}", flush=True)
        finally:
            adapter = None   # drop the loop's reference too, or the next model loads beside this one
            holder.release()

    pairs = cons.image_pairs(rows)
    by_asset: dict[str, list] = defaultdict(list)
    for pair in pairs:
        by_asset[str(pair[0]["asset_category"])].append(pair)
    leaderboard, per_asset = [], []
    for (model_key, condition), choice in preds.items():
        leaderboard.append({"model_key": model_key, "condition": condition, **cons.consistency_row(pairs, choice)})
        for asset, subset in sorted(by_asset.items()):
            per_asset.append({"model_key": model_key, "condition": condition, "asset_category": asset,
                              **cons.consistency_row(subset, choice)})
    found = {k: v for k, v in preds.items()}
    sig = cons.significance_rows(pairs, [], found, args.models, COMPARISONS, n_boot=args.n_boot)
    sig = [r for r in sig if r["metric"] not in ("q3_accuracy", "q1q3_contradiction")]

    cons.write_csv(args.out_dir / "leaderboard.csv", leaderboard)
    cons.write_csv(args.out_dir / "metrics_by_asset.csv", per_asset)
    cons.write_csv(args.out_dir / "significance.csv", sig)
    (args.out_dir / "run_status.json").write_text(json.dumps(statuses, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    meta = {
        "protocol": "gold_v2_dev_baselines", "prompt_version": PROMPT_VERSION, "dev_qa": args.dev_qa.name,
        "images": len(pairs), "questions": len(rows), "models": list(args.models), "conditions": list(args.conditions),
        "prompt_sources": {"image_only": "run_gold_baselines.prompt_for, [target]",
                           "normal_ref_fixed": "prompt_for_condition normal_reference, [fixed normal, target]",
                           "normal_ref_retrieved": "prompt_for_condition normal_reference, [retrieved normal, target]"},
        "significance": f"image-level paired bootstrap, {args.n_boot} resamples, two-sided 95% CI",
        "purpose": "method selection on the development set; paper numbers come from gold v2",
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
    errors = sum(s["row_errors"] for s in statuses)
    print(json.dumps(statuses, ensure_ascii=False, indent=2))
    if errors:
        print(f"WARNING: {errors} rows failed; re-run the same cell to retry only those rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
