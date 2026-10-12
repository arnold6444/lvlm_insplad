#!/usr/bin/env python3
"""Fresh full-image vs matched-part inference on manually annotated dev pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import dev_part_crop_annotation as annotation
import run_dev_prompt_diagnostic as diag

CONDITIONS = ("whole_image", "part_crop")


def crop_pair(pair, boxes, out_dir):
    from PIL import Image

    images = []
    for role in ("reference", "target"):
        original = Path(pair[role]["path"])
        if hashlib.sha256(original.read_bytes()).hexdigest() != pair[role]["sha256"]:
            raise ValueError("source image changed since annotation")
        key = diag.digest([pair["pair_identity"], role, boxes[role+"_box"]])
        output = Path(out_dir) / f"{key}.png"
        with Image.open(original) as image:
            if list(image.size) != pair[role]["size"]:
                raise ValueError("source image dimensions changed")
            crop = image.convert("RGB").crop(annotation.box_pixels(boxes[role+"_box"], image.size))
        # Re-create exactly the annotated crop, never reuse a stale cached crop.
        output.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output, format="PNG")
        images.append(output)
    return images


def main(argv=None, *, loader=None, infer=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--stage1-dir", type=Path, required=True)
    p.add_argument("--models", nargs="+", choices=list(diag.dev.MODELS), default=["internvl3_5_8b"])
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)
    if not args.workspace.name.startswith("gold_v2_part_crop_diagnostic"):
        raise ValueError("use a separate diagnostic workspace")
    if len(args.models) != len(set(args.models)):
        raise ValueError("duplicate models")
    all_rows = diag.agent.read_jsonl(args.workspace / "subset_qa.jsonl")
    if not all_rows or any(r.get("protocol") != "gold_v2_dev" for r in all_rows):
        raise ValueError("only dev rows are allowed")
    pairs = json.loads((args.workspace / "pairs.json").read_text(encoding="utf-8"))
    if {r["core_id"] for r in all_rows} != {pair["core_id"] for pair in pairs}:
        raise ValueError("pair/QA mismatch")
    session = annotation.PartSession(pairs, args.workspace / "annotations.json")
    eligible, excluded = session.completed()
    by_core = {p["core_id"]: p for p in eligible}
    rows = [r for r in all_rows if r["core_id"] in by_core]
    ids = {r["id"] for r in rows}
    if len(ids) != len(rows) or any(sum(q["core_id"] == p["core_id"] and q["question_type"] == qt for q in rows) != 1
                                    for p in eligible for qt in (diag.agent.Q1, diag.agent.Q2)):
        raise ValueError("each included image needs exactly Q1 and Q2")
    first = {m: diag.agent.load_answers(args.stage1_dir / m / "normal_ref_retrieved_diffdate.jsonl")[0] for m in args.models}
    if any(first[m].get(r["id"]) not in r["options"] for m in args.models for r in rows):
        raise ValueError("missing/invalid old baseline answer")
    identity = diag.digest({"version": 1, "rows": rows, "models": args.models,
                            "pairs": [{"core_id": pair["core_id"], "identity": pair["pair_identity"],
                                       "reference_box": session.data[pair["core_id"]]["reference_box"],
                                       "target_box": session.data[pair["core_id"]]["target_box"]} for pair in eligible],
                            "excluded": excluded, "first": first,
                            "prompts": [diag.dev.build_prompt(r, "normal_ref_retrieved_diffdate") for r in rows],
                            "specs": {m: diag.agent.base.MODEL_REGISTRY[m] for m in args.models}})
    out = args.workspace / "inference"
    meta_path = out / "run_metadata.json"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8"))["identity"] != identity:
        raise ValueError("annotation/inputs changed after inference; use a new workspace to preserve the old run")
    out.mkdir(parents=True, exist_ok=True)
    image_inputs = {}
    for pair in eligible:
        cropped = crop_pair(pair, session.data[pair["core_id"]], args.workspace / "_crops")
        image_inputs[(pair["core_id"], "whole_image")] = [Path(pair[role]["path"]) for role in ("reference", "target")]
        image_inputs[(pair["core_id"], "part_crop")] = cropped
    coverage = {"selected_images": len(pairs), "included_images": len(eligible), "excluded_images": len(excluded),
                "excluded": excluded,
                "included_by_asset_gt": [{"asset": asset, "gt": gt, "images": n} for (asset, gt), n in
                                          sorted(Counter((r["asset_category"], r["anomaly_gt"]) for r in rows if r["question_type"] == diag.agent.Q1).items())]}
    annotation.atomic_json(out / "coverage.json", coverage)
    meta = {"protocol": "dev_part_crop_diagnostic", "identity": identity, "conditions": CONDITIONS,
            "models": args.models, "benchmark_commit": diag.agent.git_commit(), "gpu": None,
            "model_specs": {m: diag.agent.base.MODEL_REGISTRY[m] for m in args.models},
            "questions": len(rows), "images": len(eligible), "manual_annotation": True,
            "prompt": "identical original baseline prompt in both conditions; no previous answers, review or definitions",
            "decoding": "existing infer_hf; InternVL greedy and same per-image tile budget",
            "claim_limit": "manual part diagnostic on included dev pairs, not automatic localization or gold performance",
            "selection": "none; no selection.csv or frozen_config.json"}
    annotation.atomic_json(meta_path, meta)
    diag.agent.base.write_jsonl(out / "paired_qa.jsonl", rows)
    diag.agent.base.write_jsonl(out / "prompts.jsonl", [{"id": r["id"], "prompt": diag.dev.build_prompt(r, "normal_ref_retrieved_diffdate")} for r in rows])
    from PIL import Image
    from insplad_benchmark.hf_adapters import _dynamic_preprocess

    image_manifest = []
    for (core, condition), paths in image_inputs.items():
        for role, path in zip(("reference", "target"), paths):
            with Image.open(path) as image:
                image_manifest.append({"core_id": core, "condition": condition, "role": role, "path": str(path),
                                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                       "width": image.width, "height": image.height,
                                       "expected_internvl_patches": len(_dynamic_preprocess(image, max_num=3))})
    diag.agent.cons.write_csv(out / "image_manifest.csv", image_manifest)
    if infer is None:
        from insplad_benchmark.hf_adapters import infer_hf as infer
    found, statuses, leaderboard, answers = {}, [], [], []
    for model in args.models:
        holder = diag.agent.base.LazyHFAdapter(diag.agent.base.MODEL_REGISTRY[model], loader=loader)
        try:
            for condition in CONDITIONS:
                path = out / "predictions" / model / f"{condition}.jsonl"
                old = {r["id"]: r for r in (diag.agent.read_jsonl(path) if path.exists() else [])}
                done, called, reused = [], 0, 0
                for row in rows:
                    qid = row["id"]
                    rec = old.get(qid)
                    if rec and rec.get("identity") == identity and not rec.get("error") and rec.get("output", "").strip():
                        reused += 1
                    else:
                        images = image_inputs[(row["core_id"], condition)]
                        prompt = diag.dev.build_prompt(row, "normal_ref_retrieved_diffdate")
                        output, error, start = "", "", time.time()
                        try:
                            output = str(infer(holder.get(), images, prompt))
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                        rec = {"id": qid, "identity": identity, "model_key": model, "condition": condition,
                               "images": list(map(str, images)), "output": output, "error": error,
                               "predicted_choice": diag.agent.base.parse_choice(output, set(row["options"])),
                               "latency_s": round(time.time()-start, 3)}
                        called += 1
                    done.append(rec)
                    if args.checkpoint_every > 0 and len(done) % args.checkpoint_every == 0:
                        done_ids = {r["id"] for r in done}
                        diag.agent.base.write_jsonl(path, done+[r for i, r in old.items() if i not in done_ids])
                        print(f"[{model}/{condition}] {len(done)}/{len(rows)} saved", flush=True)
                diag.agent.base.write_jsonl(path, done)
                errors = sum(bool(r["error"]) for r in done)
                statuses.append({"model_key": model, "condition": condition, "questions": len(rows),
                                 "called_this_run": called, "reused_predictions": reused, "row_errors": errors,
                                 "invalid_answers": sum(r["predicted_choice"] is None for r in done),
                                 "status": "complete" if errors == 0 else "has_errors"})
                final = {r["id"]: r["predicted_choice"] for r in done}
                for row in rows:
                    choice = final[row["id"]]
                    answers.append({"model_key": model, "condition": condition, "id": row["id"],
                                    "asset_category": row["asset_category"], "question_type": row["question_type"],
                                    "answer": row["answer"], "old_baseline_choice": first[model][row["id"]],
                                    "predicted_choice": choice, "predicted_text": row["options"].get(choice, "INVALID"),
                                    "correct": choice == row["answer"]})
                if errors == 0:
                    found[(model, condition)] = final
                    leaderboard.append({"model_key": model, "condition": condition,
                                        **diag.agent.summary_row(rows, first[model], final, set(), {}),
                                        "inference_calls": len(done), "latency_s": round(sum(r["latency_s"] for r in done), 3),
                                        "old_baseline_agreement": sum(final[i] == first[model][i] for i in ids)/len(ids)})
        finally:
            holder.release()
    sig = diag.agent.cons.significance_rows(diag.agent.cons.image_pairs(rows), [], found, args.models,
                                           [("whole_image", "part_crop")], n_boot=args.n_boot)
    sig = [r for r in sig if r["metric"] not in ("q3_accuracy", "q1q3_contradiction")]
    for name, records in (("leaderboard.csv", leaderboard), ("answers.csv", answers), ("significance.csv", sig)):
        diag.agent.cons.write_csv(out / name, records)
    annotation.atomic_json(out / "run_status.json", statuses)
    meta["gpu"] = diag.agent.gpu_info()  # after inference has loaded torch, not before
    annotation.atomic_json(meta_path, meta)
    print(json.dumps(statuses, indent=2), flush=True)
    return 1 if any(s["row_errors"] for s in statuses) else 0


if __name__ == "__main__":
    raise SystemExit(main())
