#!/usr/bin/env python3
"""Evaluate the MMAD-style gold-v2 location set under the four main conditions.

The location questions (``scripts/build_gold_v2_loc_mmad.py``) are asked with exactly the prompt builders and image
orders of the original runs, so location accuracy lines up with the existing Q1/Q2 results:

- ``image_only``   : ``run_gold_baselines.prompt_for``, images ``[target]``
- ``normal_ref``   : ``run_gold_v2_reference_ablation.prompt_for_condition(..., "normal_reference")``, ``[normal, target]``
- ``expert_score`` : ``run_gold_v2_vision_expert_full.prompt(..., "score", expert)``, ``[target]``
- ``combined``     : ``run_gold_v2_reference_expert.build_prompt(..., "score", expert)``, ``[normal, target]``

The expert score percentile is recomputed over the full 246-image cohort exactly as in the vision-expert run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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
import run_gold_v2_baselines as v2  # noqa: E402
import run_gold_v2_reference_ablation as ref  # noqa: E402
import run_gold_v2_reference_expert as rx  # noqa: E402
import run_gold_v2_vision_expert_full as vx  # noqa: E402

MODELS = ("qwen3_5_4b", "internvl3_5_8b", "minicpm_v_4_5", "llava_onevision_7b")
CONDITIONS = ("image_only", "normal_ref", "expert_score", "combined")
PROMPT_VERSION = 1


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_prompt(row: Mapping, condition: str, expert: str, percentile: float | None) -> str:
    if condition == "image_only":
        return base.prompt_for(dict(row))
    if condition == "normal_ref":
        return ref.prompt_for_condition(dict(row), "normal_reference")
    if condition == "expert_score":
        return vx.prompt(row, "score", expert, float(percentile))
    if condition == "combined":
        return rx.build_prompt(row, "score", expert, float(percentile))
    raise ValueError(condition)


def images_for(condition: str, target: Path, normal: Path) -> list[Path]:
    return [normal, target] if condition in ("normal_ref", "combined") else [target]


def expert_percentiles(manifest_rows: Sequence[Mapping[str, str]], cores: set[str], expert: str) -> dict[str, float]:
    selected = [dict(r) for r in manifest_rows if r.get("expert") == expert and str(r.get("query_id")) in cores]
    if len({str(r["query_id"]) for r in selected}) != len(cores):
        raise RuntimeError(f"{expert}: manifest covers {len(selected)} of {len(cores)} gold images")
    vx.add_score_percentiles(selected)
    return {str(r["query_id"]): float(r["score_percentile_full"]) for r in selected}


def mcnemar_exact(base_ok: Sequence[bool], new_ok: Sequence[bool]) -> tuple[int, int, float]:
    b = sum(n and not o for o, n in zip(base_ok, new_ok))
    c = sum(o and not n for o, n in zip(base_ok, new_ok))
    n = b + c
    if n == 0:
        return b, c, 1.0
    return b, c, min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def reusable(old: Mapping | None, model_key: str, condition: str) -> bool:
    return bool(old and not old.get("error") and str(old.get("output", "")).strip()
                and old.get("model_key") == model_key and old.get("condition") == condition
                and old.get("prompt_version") == PROMPT_VERSION)


def main(argv: Sequence[str] | None = None, *, load_qa: Callable[[Path], list[dict]] = v2.load_gold_qa_v2,
         loader: Callable | None = None, infer: Callable | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--loc-qa", type=Path, required=True, help="localization_mmad_qa_*.jsonl")
    p.add_argument("--qa-dir", type=Path, required=True, help="gold_v2_final (for the 246-image score cohort)")
    p.add_argument("--sup-zip", type=Path, required=True)
    p.add_argument("--unsup-zip", type=Path, required=True)
    p.add_argument("--roi-manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    p.add_argument("--expert", default="patchcore")
    p.add_argument("--checkpoint-every", type=int, default=20)
    args = p.parse_args(argv)

    if infer is None:
        from insplad_benchmark.hf_adapters import infer_hf as infer  # noqa: N813
    rows = read_jsonl(args.loc_qa)
    if not rows or any(r.get("protocol") != "gold_v2_loc_mmad" for r in rows):
        raise RuntimeError("expected a gold_v2_loc_mmad location set")
    gold_cores = {str(r["core_id"]) for r in load_qa(args.qa_dir)}
    pct = {}
    if {"expert_score", "combined"} & set(args.conditions):
        pct = expert_percentiles(vx.read_csv(args.roi_manifest), gold_cores, args.expert)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = base.prepare_images(rows, args.sup_zip, args.out_dir / "_images" / "target")
    normals = {m: base.safe_extract(args.unsup_zip, m, args.out_dir / "_images" / "normal")
               for m in sorted({str(r["normal_reference_image"]) for r in rows})}

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
                    percentile = pct.get(str(row["core_id"]))
                    prompt = build_prompt(row, condition, args.expert, percentile)
                    images = images_for(condition, targets[str(row["image"])], normals[str(row["normal_reference_image"])])
                    adapter = holder.get()
                    started = time.time()
                    output, error = "", ""
                    try:
                        output = str(infer(adapter, images, prompt))
                    except Exception as exc:  # retried on the next run
                        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
                        errors += 1
                    out_rows.append({
                        "id": row["id"], "core_id": row["core_id"], "model_key": model_key, "condition": condition,
                        "prompt_version": PROMPT_VERSION, "score_percentile_full": percentile, "output": output,
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

    def correct(choice: Mapping[str, str | None], subset: Sequence[Mapping]) -> list[bool]:
        return [choice.get(str(r["id"])) == r["answer"] for r in subset]

    leaderboard, by_group = [], []
    groups: dict[tuple[str, str], list[Mapping]] = defaultdict(list)
    for r in rows:
        groups[("asset_category", str(r["asset_category"]))].append(r)
        groups[("defect_gt", str(r["defect_gt"]))].append(r)
        groups[("label_basis", str(r["localization_gt"]["basis"]))].append(r)
    for model_key in args.models:
        img = preds.get((model_key, "image_only"))
        for condition in args.conditions:
            choice = preds[(model_key, condition)]
            ok = correct(choice, rows)
            item = {"model_key": model_key, "condition": condition, "questions": len(rows),
                    "loc_accuracy": round(sum(ok) / len(rows), 6), "chance": 0.25,
                    "valid_output_rate": round(sum(choice.get(str(r["id"])) is not None for r in rows) / len(rows), 6)}
            if img is not None and condition != "image_only":
                b, c, pval = mcnemar_exact(correct(img, rows), ok)
                item.update({"delta_vs_image_only": round((sum(ok) - sum(correct(img, rows))) / len(rows), 6),
                             "gain_vs_image_only": b, "loss_vs_image_only": c, "mcnemar_p_vs_image_only": round(pval, 6)})
            leaderboard.append(item)
            for (kind, value), subset in sorted(groups.items()):
                by_group.append({"model_key": model_key, "condition": condition, "group": kind, "value": value,
                                 "questions": len(subset), "loc_accuracy": round(sum(correct(choice, subset)) / len(subset), 6)})

    write_csv(args.out_dir / "leaderboard.csv", leaderboard)
    write_csv(args.out_dir / "metrics_by_group.csv", by_group)
    (args.out_dir / "run_status.json").write_text(json.dumps(statuses, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    metadata = {
        "protocol": "gold_v2_loc_mmad_eval", "prompt_version": PROMPT_VERSION, "loc_qa": args.loc_qa.name,
        "questions": len(rows), "models": list(args.models), "conditions": list(args.conditions),
        "expert": args.expert,
        "prompt_sources": {
            "image_only": "run_gold_baselines.prompt_for, [target]",
            "normal_ref": "run_gold_v2_reference_ablation.prompt_for_condition normal_reference, [normal, target]",
            "expert_score": "run_gold_v2_vision_expert_full.prompt score, [target]",
            "combined": "run_gold_v2_reference_expert.build_prompt score, [normal, target]",
        },
        "score_percentile": "within-expert, within-asset percentile over the 246-image gold cohort (transductive)",
        "test": "exact two-sided McNemar vs image_only on the same questions",
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
    errors = sum(s["row_errors"] for s in statuses)
    print(json.dumps(statuses, ensure_ascii=False, indent=2))
    if errors:
        print(f"WARNING: {errors} rows failed; re-run the same cell to retry only those rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
