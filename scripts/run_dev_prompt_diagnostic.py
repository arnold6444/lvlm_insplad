#!/usr/bin/env python3
"""Paired dev-only prompt diagnosis; never selects or freezes a gold configuration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_gold_v2_verify_agent as agent
import run_gold_v2_dev_baselines as dev

CONDITIONS = ("baseline", "review_only", "prior_only", "current_agent")
VERSION = 1
REVIEW = "Re-examine the TARGET carefully and answer this question independently."


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build_prompt(row, own_first, condition):
    baseline = dev.build_prompt(row, "normal_ref_retrieved_diffdate")
    if condition == "baseline":
        return baseline
    if condition == "current_agent":
        return agent.build_prompt(row, own_first, "reask")
    if condition == "review_only":
        extra = REVIEW
    elif condition == "prior_only":
        if own_first not in row["options"]:
            raise ValueError("diagnosis requires a valid own stage-1 answer")
        extra = f'The previous answer to this question was "{own_first}. {row["options"][own_first]}".'
    else:
        raise ValueError(condition)
    return baseline.replace("\n\nQuestion:", f"\n\n{extra}\n\nQuestion:", 1)


def select_subset(rows, n_images=40, seed=42):
    """Round-robin asset/GT/Q1 option-order strata, without using model outputs."""
    if not 1 <= n_images <= 80:
        raise ValueError("use 1..80 dev images; this runner is a small diagnosis")
    by_core = defaultdict(dict)
    for row in rows:
        if row.get("protocol") != "gold_v2_dev":
            raise ValueError("only gold_v2_dev questions are allowed")
        if row["question_type"] not in (agent.Q1, agent.Q2):
            continue
        if row["question_type"] in by_core[row["core_id"]]:
            raise ValueError("duplicate question in a core")
        by_core[row["core_id"]][row["question_type"]] = row
    strata = defaultdict(list)
    for core, qs in by_core.items():
        if set(qs) != {agent.Q1, agent.Q2} or qs[agent.Q1]["image"] != qs[agent.Q2]["image"]:
            raise ValueError("each image needs its own Q1 and Q2")
        q = qs[agent.Q1]
        strata[(q["asset_category"], q["anomaly_gt"], tuple(q["options"].items()))].append(core)
    if len(by_core) < n_images:
        raise ValueError("not enough dev images")
    for cores in strata.values():
        cores.sort(key=lambda c: hashlib.sha256(f"{seed}:{c}".encode()).hexdigest())
    assets = sorted({k[0] for k in strata})
    keys = {a: sorted(k for k in strata if k[0] == a) for a in assets}
    chosen = []
    cursor = defaultdict(int)
    while len(chosen) < n_images:
        for asset in assets:
            available = [k for k in keys[asset] if strata[k]]
            if not available:
                continue
            key = available[cursor[asset] % len(available)]
            cursor[asset] += 1
            chosen.append(strata[key].pop(0))
            if len(chosen) == n_images:
                break
    return [by_core[c][q] for c in chosen for q in (agent.Q1, agent.Q2)]


def main(argv=None, *, loader=None, infer=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dev-qa", "stage1-dir", "refs-csv", "sup-zip", "unsup-zip", "out-dir"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--models", nargs="+", choices=list(dev.MODELS), default=["internvl3_5_8b"])
    p.add_argument("--n-images", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)
    # Protect previous results even if the caller gives the wrong output directory.
    if not args.out_dir.name.startswith("gold_v2_prompt_diagnostic"):
        raise ValueError("output must be a separate gold_v2_prompt_diagnostic* directory")
    if len(args.models) != len(set(args.models)):
        raise ValueError("duplicate models")
    rows = select_subset(agent.read_jsonl(args.dev_qa), args.n_images, args.seed)
    ids = {r["id"] for r in rows}
    with args.refs_csv.open(encoding="utf-8-sig", newline="") as h:
        refs = {r["image"]: r["retrieved_diffdate"] for r in csv.DictReader(h)}
    first = {}
    for model in args.models:
        first[model] = agent.load_answers(args.stage1_dir / model / "normal_ref_retrieved_diffdate.jsonl")[0]
        if any(first[model].get(r["id"]) not in r["options"] for r in rows):
            raise ValueError("stage1 must cover every selected question with a valid choice")
    if any(not refs.get(r["image"]) for r in rows):
        raise ValueError("missing diffdate reference")
    prompt_audit = [{"model_key": m, "condition": c, "id": r["id"],
                     "own_first_choice": first[m][r["id"]], "prompt": build_prompt(r, first[m][r["id"]], c)}
                    for m in args.models for c in CONDITIONS for r in rows]
    identity = digest({"version": VERSION, "seed": args.seed, "rows": rows, "models": args.models,
                       "refs": {r["image"]: refs[r["image"]] for r in rows}, "prompts": prompt_audit,
                       "model_specs": {m: agent.base.MODEL_REGISTRY[m] for m in args.models}})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.out_dir / "run_metadata.json"
    if meta_path.exists() and json.loads(meta_path.read_text())["identity"] != identity:
        raise ValueError("output already contains different inputs; use a new diagnostic folder")
    meta = {"protocol": "dev_prompt_diagnostic", "version": VERSION, "identity": identity,
            "models": args.models, "images": args.n_images, "questions": len(rows), "seed": args.seed,
            "model_specs": {m: agent.base.MODEL_REGISTRY[m] for m in args.models},
            "decoding": "same infer_hf as baseline; InternVL greedy, do_sample=False",
            "conditions": CONDITIONS, "benchmark_commit": agent.git_commit(), "gpu": agent.gpu_info(),
            "selection": "diagnosis only; no selection.csv or frozen_config.json",
            "sample": "round-robin asset / GT / Q1 option order; no model-output-based selection",
            "caveat": "current_agent also omits the baseline hidden-metadata sentence; not a pure 2x2 factorial"}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    agent.base.write_jsonl(args.out_dir / "subset_qa.jsonl", rows)
    agent.base.write_jsonl(args.out_dir / "prompts.jsonl", prompt_audit)
    targets = agent.base.prepare_images(rows, args.sup_zip, args.out_dir / "_images" / "target")
    normals = {ref: agent.base.safe_extract(args.unsup_zip, ref, args.out_dir / "_images" / "reference")
               for ref in sorted({refs[r["image"]] for r in rows})}
    from PIL import Image
    from insplad_benchmark.hf_adapters import _dynamic_preprocess
    manifest = []
    for member, path in list(targets.items()) + list(normals.items()):
        with Image.open(path) as image:
            manifest.append({"member": member, "path": str(path), "width": image.width, "height": image.height,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "internvl_patches_two_images": len(_dynamic_preprocess(image, max_num=3))})
    agent.cons.write_csv(args.out_dir / "image_manifest.csv", manifest)
    meta["image_identity"] = digest([{k: r[k] for k in ("member", "sha256")} for r in manifest])
    # Per-prediction identity also binds actual extracted image bytes.
    identity = digest([identity, meta["image_identity"]])
    meta["prediction_identity"] = identity
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if infer is None:
        from insplad_benchmark.hf_adapters import infer_hf as infer
    found, statuses, leaderboard, answers = {}, [], [], []
    for model in args.models:
        holder = agent.base.LazyHFAdapter(agent.base.MODEL_REGISTRY[model], loader=loader)
        try:
            for condition in CONDITIONS:
                path = args.out_dir / "predictions" / model / f"{condition}.jsonl"
                old = {r["id"]: r for r in (agent.read_jsonl(path) if path.exists() else [])}
                done = []
                called = reused = 0
                for row in rows:
                    qid = row["id"]
                    record = old.get(qid)
                    if record and record.get("identity") == identity and not record.get("error") and record.get("output", "").strip():
                        reused += 1
                    else:
                        prompt = build_prompt(row, first[model][qid], condition)
                        images = [normals[refs[row["image"]]], targets[row["image"]]]
                        start = time.time()
                        output, error = "", ""
                        try:
                            output = str(infer(holder.get(), images, prompt))
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                        record = {"id": qid, "identity": identity, "model_key": model, "condition": condition,
                                  "own_first_choice": first[model][qid], "output": output, "error": error,
                                  "predicted_choice": agent.base.parse_choice(output, set(row["options"])),
                                  "latency_s": round(time.time() - start, 3), "images": list(map(str, images))}
                        called += 1
                    done.append(record)
                    if args.checkpoint_every > 0 and len(done) % args.checkpoint_every == 0:
                        agent.base.write_jsonl(path, done + [r for i, r in old.items() if i not in {d["id"] for d in done}])
                        print(f"[{model}/{condition}] {len(done)}/{len(rows)} saved", flush=True)
                agent.base.write_jsonl(path, done)
                errors = sum(bool(r["error"]) for r in done)
                invalid = sum(r["predicted_choice"] is None for r in done)
                statuses.append({"model_key": model, "condition": condition, "questions": len(rows),
                                 "called_this_run": called, "reused_predictions": reused, "row_errors": errors,
                                 "invalid_answers": invalid, "status": "complete" if errors == 0 else "has_errors"})
                final = {r["id"]: r["predicted_choice"] for r in done}
                for r in rows:
                    choice = final[r["id"]]
                    answers.append({"model_key": model, "condition": condition, "id": r["id"],
                                    "asset_category": r["asset_category"], "question_type": r["question_type"],
                                    "answer": r["answer"], "stage1_choice": first[model][r["id"]],
                                    "predicted_choice": choice, "correct": choice == r["answer"],
                                    "predicted_text": r["options"].get(choice, "INVALID")})
                if errors == 0:
                    found[(model, condition)] = final
                    leaderboard.append({"model_key": model, "condition": condition,
                                        **agent.summary_row(rows, first[model], final, ids,
                                                            {r["id"]: r["latency_s"] for r in done}),
                                        "stage1_agreement": sum(final[i] == first[model][i] for i in ids) / len(ids)})
        finally:
            holder.release()
    sig = agent.cons.significance_rows(agent.cons.image_pairs(rows), [], found, args.models,
                                      [("baseline", c) for c in CONDITIONS[1:]], n_boot=args.n_boot)
    sig = [r for r in sig if r["metric"] not in ("q3_accuracy", "q1q3_contradiction")]
    for name, records in (("leaderboard.csv", leaderboard), ("answers.csv", answers), ("significance.csv", sig)):
        agent.cons.write_csv(args.out_dir / name, records)
    (args.out_dir / "run_status.json").write_text(json.dumps(statuses, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(statuses, indent=2), flush=True)
    return 1 if any(s["row_errors"] for s in statuses) else 0


if __name__ == "__main__":
    raise SystemExit(main())
