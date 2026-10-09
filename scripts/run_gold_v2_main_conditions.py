#!/usr/bin/env python3
"""Re-run gold-v2 main conditions with the current adapters, and add retrieved-reference conditions.

Used for (a) the MiniCPM greedy re-run of the four main conditions (MiniCPM had sampled at temperature 0.7) and
(b) the retrieved normal-reference baselines for every model. All prompts and image orders come from the original
runs (via ``run_gold_v2_loc_mmad_eval.build_prompt``):

- ``image_only``                     [target]
- ``normal_reference``               [fixed per-asset normal, target]
- ``vx:patchcore:score``             [target] + PatchCore percentile sentence
- ``ref+patchcore:score``            [fixed normal, target] + PatchCore percentile sentence
- ``normal_ref_retrieved``           [DINOv2 top-1 normal, target]          (normal-reference prompt)
- ``normal_ref_retrieved_diffdate``  [DINOv2 top-1 from another flight, target] (normal-reference prompt)

Every question (Q1, Q2, Q3) is asked. Metrics use the consistency definitions (Q1 F1, Q2 macro-F1, Q3 accuracy,
Q1-Q2 / Q1-Q3 contradiction, joint accuracy, FDR).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_gold_baselines as base  # noqa: E402
import run_gold_v2_baselines as v2  # noqa: E402
import run_gold_v2_consistency_metrics as cons  # noqa: E402
import run_gold_v2_loc_mmad_eval as loc  # noqa: E402
import run_gold_v2_vision_expert_full as vx  # noqa: E402

CONDITIONS = {
    # condition -> (prompt builder key in loc.build_prompt, reference kind, file slug)
    "image_only": ("image_only", None, "image_only"),
    "normal_reference": ("normal_ref", "fixed", "normal_reference"),
    "vx:patchcore:score": ("expert_score", None, "vx_patchcore_score"),
    "ref+patchcore:score": ("combined", "fixed", "ref_patchcore_score"),
    "normal_ref_retrieved": ("normal_ref", "retrieved_any", "normal_ref_retrieved"),
    "normal_ref_retrieved_diffdate": ("normal_ref", "retrieved_diffdate", "normal_ref_retrieved_diffdate"),
}
SLUG_TO_CONDITION = cons.RERUN_SLUG_TO_CONDITION
assert SLUG_TO_CONDITION == {slug: cond for cond, (_, _, slug) in CONDITIONS.items()}
PROMPT_VERSION = 1


def read_refs(path: Path | None) -> dict[str, dict]:
    if not path:
        return {}
    with Path(path).open(newline="", encoding="utf-8-sig") as h:
        return {r["image"]: r for r in csv.DictReader(h)}


def images_for(condition: str, row: Mapping, target: Path, refs: Mapping[str, Path], retrieved: Mapping[str, dict]) -> list[Path]:
    kind = CONDITIONS[condition][1]
    if kind is None:
        return [target]
    if kind == "fixed":
        return [refs[str(row["normal_reference_image"])], target]
    member = retrieved[str(row["image"])][kind]
    return [refs[member], target]


def reusable(old: Mapping | None, model_key: str, condition: str) -> bool:
    return bool(old and not old.get("error") and str(old.get("output", "")).strip()
                and old.get("model_key") == model_key and old.get("condition") == condition
                and old.get("prompt_version") == PROMPT_VERSION)


def main(argv: Sequence[str] | None = None, *, load_qa: Callable[[Path], list[dict]] = v2.load_gold_qa_v2,
         loader: Callable | None = None, infer: Callable | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--qa-dir", type=Path, required=True)
    p.add_argument("--sup-zip", type=Path, required=True)
    p.add_argument("--unsup-zip", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--conditions", nargs="+", required=True, choices=list(CONDITIONS))
    p.add_argument("--roi-manifest", type=Path, help="needed for the PatchCore score conditions")
    p.add_argument("--refs-csv", type=Path, help="retrieved_refs.csv, needed for retrieved conditions")
    p.add_argument("--expert", default="patchcore")
    p.add_argument("--checkpoint-every", type=int, default=20)
    args = p.parse_args(argv)

    if infer is None:
        from insplad_benchmark.hf_adapters import infer_hf as infer  # noqa: N813
    rows = load_qa(args.qa_dir)
    cores = {str(r["core_id"]) for r in rows}
    pct: dict[str, float] = {}
    if {"vx:patchcore:score", "ref+patchcore:score"} & set(args.conditions):
        if not args.roi_manifest:
            raise RuntimeError("--roi-manifest is required for PatchCore score conditions")
        pct = loc.expert_percentiles(vx.read_csv(args.roi_manifest), cores, args.expert)
    retrieved = read_refs(args.refs_csv)
    if {"normal_ref_retrieved", "normal_ref_retrieved_diffdate"} & set(args.conditions):
        missing = sorted({str(r["image"]) for r in rows} - set(retrieved))
        if missing:
            raise RuntimeError(f"--refs-csv lacks {len(missing)} gold images, e.g. {missing[:2]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = base.prepare_images(rows, args.sup_zip, args.out_dir / "_images" / "target")
    members = {str(r["normal_reference_image"]) for r in rows}
    for item in retrieved.values():
        members |= {item["retrieved_any"], item["retrieved_diffdate"]}
    refs = {m: base.safe_extract(args.unsup_zip, m, args.out_dir / "_images" / "normal") for m in sorted(members)}

    statuses: list[dict] = []
    for model_key in args.models:
        holder = base.LazyHFAdapter(base.MODEL_REGISTRY[model_key], loader=loader)
        try:
            for condition in args.conditions:
                print(f"\n=== {model_key} / {condition} ===", flush=True)
                pred_path = args.out_dir / "predictions" / model_key / f"{CONDITIONS[condition][2]}.jsonl"
                existing = {str(r["id"]): r for r in (loc.read_jsonl(pred_path) if pred_path.is_file() else [])}
                out_rows: list[dict] = []
                called = reused = errors = 0
                for row in rows:
                    old = existing.get(str(row["id"]))
                    if reusable(old, model_key, condition):
                        out_rows.append(old)
                        reused += 1
                        continue
                    percentile = pct.get(str(row["core_id"]))
                    prompt = loc.build_prompt(row, CONDITIONS[condition][0], args.expert, percentile)
                    images = images_for(condition, row, targets[str(row["image"])], refs, retrieved)
                    adapter = holder.get()
                    started = time.time()
                    output, error = "", ""
                    try:
                        output = str(infer(adapter, images, prompt))
                    except Exception as exc:  # retried on the next run
                        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
                        errors += 1
                    out_rows.append({
                        "id": row["id"], "core_id": row["core_id"], "question_type": row["question_type"],
                        "model_key": model_key, "condition": condition, "prompt_version": PROMPT_VERSION,
                        "reference_image": str(images[0]) if len(images) == 2 else "", "output": output,
                        "predicted_choice": base.parse_choice(output, {str(k).upper() for k in row["options"]}),
                        "latency_s": round(time.time() - started, 3), "error": error,
                    })
                    called += 1
                    if args.checkpoint_every > 0 and called % args.checkpoint_every == 0:
                        base.write_jsonl(pred_path, out_rows)
                        print(f"[{model_key}/{condition}] {len(out_rows)}/{len(rows)} checkpoint", flush=True)
                base.write_jsonl(pred_path, out_rows)
                statuses.append({"model_key": model_key, "condition": condition, "called_this_run": called,
                                 "reused_predictions": reused, "row_errors": errors,
                                 "status": "complete" if errors == 0 else "has_errors"})
                if errors:
                    first = next(r["error"] for r in out_rows if r.get("error"))
                    print(f"[{model_key}/{condition}] {errors} errors, first: {first[:300]}", flush=True)
        finally:
            adapter = None   # drop the loop's reference too, or the next model loads beside this one
            holder.release()

    # leaderboard over everything present in the output folder (earlier runs included)
    pairs, loc_pairs = cons.image_pairs(rows), cons.q3_pairs(rows)
    leaderboard = []
    for model_dir in sorted((args.out_dir / "predictions").glob("*")):
        for f in sorted(model_dir.glob("*.jsonl")):
            cond = SLUG_TO_CONDITION.get(f.stem)
            if cond is None:
                continue
            choice = {str(r["id"]): r.get("predicted_choice") for r in loc.read_jsonl(f)}
            if any(str(q["id"]) not in choice for pair in pairs for q in pair):
                continue
            leaderboard.append({"model_key": model_dir.name, "condition": cond,
                                **cons.consistency_row(pairs, choice), **cons.q3_row(loc_pairs, choice)})
    cons.write_csv(args.out_dir / "leaderboard.csv", leaderboard)
    status_path = args.out_dir / "run_status.json"
    old_status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else []
    keys = {(s["model_key"], s["condition"]) for s in statuses}
    merged = [s for s in old_status if (s["model_key"], s["condition"]) not in keys] + statuses
    status_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    meta = {
        "protocol": "gold_v2_main_rerun", "prompt_version": PROMPT_VERSION,
        "purpose": "MiniCPM greedy re-run of the main conditions (MiniCPM previously sampled at T=0.7) and "
                   "retrieved normal-reference baselines (same-flight candidates kept / excluded)",
        "decoding": "greedy for every model (MiniCPM: sampling=False, num_beams=1, repetition_penalty=1.0)",
        "conditions": {c: {"prompt": CONDITIONS[c][0], "reference": CONDITIONS[c][1]} for c in CONDITIONS},
        "refs_csv": args.refs_csv.name if args.refs_csv else None,
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
